#include <string>
#include <sstream>
#include <iostream>
#include <fstream>
#include "NetworkBCPNN.h"
#include "NetworkKussul.h"
#include "NetworkCL.h"
#include "NetworkMDS.h"
#include "NetworkMI.h"
#include "NetworkVQ.h"
#include "Network.h"

#include "NetworkTemporal3.h"
#include "NetworkTriesch.h"
#include "NetworkFoldiak.h"

//using namespace std;

void NetworkTemporal3::NetworkSetupStructure()
{
	ofstream wout;
	wout.open("time_start.txt",ios::trunc);
	wout << "Timing started!";
	wout.close();
	m_dataAssigner=new SequentialAssignmentStrategy();
	m_architecture = PC;
	m_nrInputHypercolumns = 13;
	m_nrInputRateUnits = 33;
	m_verbose=true;
	// layer 2
	m_nrHypercolumns2 = 20;//24;//96;
	m_nrRateUnits2 = 33;//100;
	int m_mdsSize=15;

	// Structures setting up everything to calculate mutual information (MI)/correlations (Pearson) on input data + multi-dimensional scaling (MDS) + vector quantization (VQ)
	m_structureInput = new StructureMIMDSVQ(); 

	m_layerInput = new PopulationColumns(this,m_nrInputHypercolumns,m_nrInputRateUnits,PopulationColumns::Graded,MPIDistribution::ParallelizationHypercolumns); // input
	m_structureInput->SetMDSDimension(m_mdsSize); // Number of dimensions in multi-dimensional scaling matrix (Size input x dimensions)
	m_structureInput->SetMDSMeasure(true);
	
	m_structureInput->SetupStructure(this,m_layerInput,m_nrHypercolumns2,m_nrRateUnits2, true);
	m_layer2 = m_structureInput->GetLayer(1);
	//PopulationModifier* wta = new WTA();
	//m_layer2->AddPopulationModifier(wta);
	//m_structureInput->MDSHypercolumns()->SetUseThreshold(true,0.9);
	m_structureInput->VQ()->GetCSL()->SetEta(0.001,true);
	//m_structureInput->VQ()->SetNrOverlaps(8);
	m_structureInput->MDSHypercolumns()->SetUpdateSize(2e-3);

	m_structureInput->MDSHypercolumns()->SwitchOnOff(false);
	m_structureInput->MDS()->SwitchOnOff(false);
	m_structureInput->VQ()->SwitchOnOff(false);
	m_structureInput->CSLLearn()->SwitchOnOff(false);

	AddTiming(this);
	AddTiming(m_structureInput->VQ());
	AddTiming(m_structureInput->MDSHypercolumns());
	AddTiming(m_structureInput->MDS());
	AddTiming(m_layerInput);
	AddTiming(m_layer2);
	AddTiming(m_structureInput->CSLLearn());
}

void NetworkTemporal3::NetworkSetupMeters()
{
	m_structureInput->SetupMeters(this->MPIGetNodeId(),this->MPIGetNrProcs());

	m_inputMeter = new Meter("inputLayer.csv", Storage::CSV);
	m_inputMeter->AttachPopulation(m_layerInput);
	AddMeter(m_inputMeter);

	m_layer1Meter = new Meter("layer1.csv", Storage::CSV);
	m_layer1Meter->AttachPopulation(m_layer2);
	AddMeter(m_layer1Meter);
}

void NetworkTemporal3::NetworkSetupParameters()
{
	// not used
}


vector<float> NetworkTemporal3::toBinary(const vector<float>& data, int nrHc, int nrMc)
{
	vector<float> out(nrHc*nrMc);

	int currentI = 0;
	for(int i=0;i<data.size();i++)
	{
		out[currentI+data[i]] = 1.0;
		currentI+=nrMc;		
	}

	return out;
}

void NetworkTemporal3::ComputeCorrelation(const vector<vector<float> >& trainingData,  PopulationColumns* inputLayer, StructureMIMDSVQ* structure, int iters) {
	structure->Pearson()->SwitchOnOff(true);
	ifstream ifile("pcorr.csv");
	if(ifile) {
		structure->Pearson()->ReadFromCSV("pcorr.csv",m_nrInputHypercolumns*m_nrInputRateUnits);
		if(m_verbose &&this->MPIGetNodeId() == 0)
			cout<<"Correlations read from file.\n";
		structure->Pearson()->SwitchOnOff(false);
		return;
	}
	for(int i=0;i<iters;i++) {
		if(m_verbose && this->MPIGetNodeId() == 0) {
			cout<<i<<"("<<iters<<") ";
			cout.flush();
		}
		inputLayer->SetValuesAll(m_dataAssigner->prepareValues(i%trainingData.size(),trainingData));
		this->Simulate();
		cout.flush();
	}
	if(m_verbose &&this->MPIGetNodeId() == 0)
		cout<<"\n";
	//structure->Pearson()->SaveToCSV("pcorr.csv",m_nrInputHypercolumns*m_nrInputRateUnits, this->network());
	structure->Pearson()->SwitchOnOff(false);
}

void NetworkTemporal3::ComputeMDS(const vector<vector<float> >& trainingData,  PopulationColumns* inputLayer, StructureMIMDSVQ* structure, int iters) {
	structure->MDSHypercolumns()->SwitchOnOff(true);
	structure->MDS()->SwitchOnOff(true);
	for(int i=0;i<iters;i++) {
		if(m_verbose && this->MPIGetNodeId() == 0) {
			cout<<i<<"("<<iters<<") ";
			cout.flush();
		}
		inputLayer->SetValuesAll(m_dataAssigner->prepareValues(i%trainingData.size(),trainingData));
		this->Simulate();
		cout.flush();
		if(!structure->MDS()->IsOn() || !structure->MDSHypercolumns()->IsOn()) {
			structure->MDS()->SwitchOnOff(false);
			structure->MDSHypercolumns()->SwitchOnOff(false);
			break;
		}
	}
	if(m_verbose && this->MPIGetNodeId() == 0)
		cout<<"\n";
	structure->MDSHypercolumns()->SwitchOnOff(false);
	structure->MDS()->SwitchOnOff(false);
}

void NetworkTemporal3::ComputeVQ(const vector<vector<float> >& trainingData,  PopulationColumns* inputLayer, StructureMIMDSVQ* structure, int iters) {
	structure->GetLayer(1)->SwitchOnOff(true);
	structure->VQ()->SwitchOnOff(true);
	for(int i=0;i<iters;i++) {
		if(m_verbose && this->MPIGetNodeId() == 0) {
			cout<<i<<"("<<iters<<") ";
			cout.flush();
		}
		inputLayer->SetValuesAll(m_dataAssigner->prepareValues(i%trainingData.size(),trainingData));
		this->Simulate();
		cout.flush();
		if(!structure->VQ()->IsOn())
			break;
	}
	if(m_verbose && this->MPIGetNodeId() == 0)
		cout<<"\n";
	structure->VQ()->SwitchOnOff(false);
}

void NetworkTemporal3::ExtractFeatures(const vector<vector<float> >& trainingData,  PopulationColumns* inputLayer, StructureMIMDSVQ* structure, int iters) {
	structure->CSLLearn()->SwitchOnOff(true);
	structure->SetRecording(false);
	float clC = m_nrRateUnits2*2;
	int idx=0;
	for(int i=0;i<iters;i++)
	{
		inputLayer->SetValuesAll(m_dataAssigner->prepareValues(i%trainingData.size(),trainingData));
		// next time step
		this->Simulate();
		if(m_verbose && this->MPIGetNodeId()==0) {
			cout << " (" << i << "/" << iters << ") ";
			cout.flush();
			if(i==trainingData.size()-1) {
				cout << "Train past" << endl;
				cout.flush();
			}
		}
	}
	if(m_verbose && this->MPIGetNodeId()== 0)
		cout<<"\n";
	//this->RecordAll();
	structure->CSLLearn()->SwitchOnOff(false);
}

void NetworkTemporal3::TrainLayer(const vector<vector<float> >& trainingData, PopulationColumns* inputLayer, StructureMIMDSVQ* structure, int iterationsCorrs, int iterationsMDS, int iterationsVQ, int iterationsFeatures)
{
	// Training phase

	int nrTrainImages = trainingData.size();
	structure->MDSHypercolumns()->SwitchOnOff(false);
	structure->MDS()->SwitchOnOff(false);
	structure->VQ()->SwitchOnOff(false);
	structure->CSLLearn()->SwitchOnOff(false);
	structure->GetLayer(1)->SwitchOnOff(false);
	structure->SetRecording(false);

	// Semi-sequential version

	// 1. Training phase
	// 1A. Patches creation
	structure->CSLLearn()->SetMaxPatterns(nrTrainImages);
	structure->CSLLearn()->SetEta(0.001);
	// turn of response in 2nd layer during initial training phase for speed
	structure->GetLayer(1)->SwitchOnOff(false);
	ComputeCorrelation(trainingData,inputLayer,structure,iterationsCorrs);
	ComputeMDS(trainingData,inputLayer,structure,iterationsMDS);
	structure->GetLayer(1)->SwitchOnOff(true);
	if(m_verbose && this->MPIGetNodeId() == 0)
		cout << "DataPoint (patches): " << iterationsVQ <<endl;
	ComputeVQ(trainingData,inputLayer,structure,iterationsVQ);
	structure->CSLLearn()->SetMaxPatterns(nrTrainImages);
	if(m_verbose && this->MPIGetNodeId() == 0)
		cout << "DataPoint (features): " << iterationsFeatures <<endl;
	ExtractFeatures(trainingData,inputLayer,structure,iterationsFeatures);
	this->RecordAll();
}

void NetworkTemporal3::NetworkRun()
{
	int nrColors = 2;
	char* filenameTrain, *filenameTest, *filenameTrainLabels;
	int nrTrainImages = 1000;
	int nrTestImages = 1000;

	int iterationsPatches = 10;
	
	int iterationsFeatures = nrTrainImages + nrTrainImages;

	if(m_architecture == PC)
	{
		nrTrainImages = 3000;
		nrTestImages = 3000;
		//iterationsPatches = 10;
		filenameTrain="D:\\Databases\\tidigits\\trainData.csv";
		filenameTrainLabels="D:\\Databases\\tidigits\\trainDataLabels.csv";
		filenameTest="D:\\Databases\\tidigits\\trainData.csv";
		/*filenameTrain="D:\\Databases\\timit\\trainn.csv";
		filenameTrainLabels="D:\\Databases\\timit\\trainL.csv";
		filenameTest="D:\\Databases\\timit\\test.csv";
		/*filenameTrain="D:\\Databases\\MLSP\\train_birds.csv";
		filenameTrainLabels="D:\\Databases\\MLSP\\trainL_birds.csv";*/
	}
	else {
		nrTrainImages = 250000;
		nrTestImages = 250000;
		filenameTrain="/cfs/klemming/nobackup/p/paherman/Tin/data/timit/train.csv";
		filenameTrainLabels="/cfs/klemming/nobackup/p/paherman/Tin/timit/data/trainL.csv";
		filenameTest="/cfs/klemming/nobackup/p/paherman/Tin/data/timit/test.csv";
	}
	int stepsStimuliOn = 1;
	// Specify input data
	Storage storage;
	Storage storageLabels;
	storage.SetMPIParameters(this->MPIGetNodeId(),this->MPIGetNrProcs());

	vector<vector<float> > trainingData = storage.LoadDataFloatCSV(filenameTrain,nrTrainImages,true);
	vector<vector<float> > trainingLabels = storageLabels.LoadDataFloatCSV(filenameTrainLabels,nrTrainImages,true);
	vector<vector<float> > testData = storage.LoadDataFloatCSV(filenameTest,nrTestImages,true);

	vector<int> partsOfDataToUseAsInput = m_layerInput->GetMPIDistributionHypercolumns(this->MPIGetNodeId());
	vector<int> partsOfDataToUseAsOutput = vector<int>();//layer1->GetMPIDistributionHypercolumns(mpiRank);

	m_layerInput->SwitchOnOff(false);
	m_structureInput->CSLLearn()->SetMaxPatterns(trainingData.size());
	// Turn off recording during training
	m_structureInput->SetRecording(false);
	m_layerInput->SetRecording(false);
	m_layer2->SetRecording(false);
	
	// Train 1st layer
	//m_layer2->GetIncomingProjections()[1]->SwitchOnOff(false); // turn off the Projections calculating correlations in layer 2 so (change to switch off in structure by default instead) (!)
	//m_layer2->GetIncomingProjections()[2]->SwitchOnOff(false);

	int iterationsCorrs = 3*trainingData.size(); // run through x times, 5
	int iterationsMDS = 1000;//200
	int iterationsVQ = 1000;//200
	iterationsFeatures = trainingData.size()+iterationsVQ; //+nrtraindata the timesteps to run CSL (first totalNrTrainingData steps are to collect the data to run on)
	TrainLayer(trainingData,m_layerInput,m_structureInput,iterationsCorrs,iterationsMDS,iterationsVQ,iterationsFeatures);
	
	// Run through all training and test data
	m_layer2->SwitchOnOff(true);

	// turn on all recording
	m_layerInput->SetRecording(true);
	//m_layer2->SetRecording(true);

	TimingStart("RunThrough");
	for(int i=0;i<trainingData.size();i++)
	{
		if(i==1)
			m_layer2->SetRecording(true);
		m_layerInput->SetValuesAll(m_dataAssigner->prepareValues(i,trainingData));
		this->Simulate();
	}

	for(int i=0;i<testData.size();i++)
	{
		m_layerInput->SetValuesAll(m_dataAssigner->prepareValues(i,testData));
		this->Simulate();
	}
	m_layerInput->SetRecording(false);
	this->Simulate();
	TimingStop("RunThrough");
}