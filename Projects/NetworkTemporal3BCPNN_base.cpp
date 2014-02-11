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

#include "NetworkTemporal3BCPNN_base.h"
#include "NetworkTriesch.h"
#include "NetworkFoldiak.h"

//using namespace std;

void NetworkTemporal3BCPNN_base::NetworkSetupStructure()
{
	ofstream wout;
	wout.open("time_start.txt",ios::trunc);
	wout << "Timing started!";
	wout.close();
	m_dataAssigner=new SequentialAssignmentStrategy();
	m_architecture = CRAY;
	m_nrInputHypercolumns = 13;
	m_nrInputRateUnits = 33;
	m_verbose=true;

	m_nrHypercolumns3 = 1;//24;//96;
	m_nrRateUnits3 = m_nrInputHypercolumns*m_nrInputRateUnits;//100;
	int m_mdsSize=15;

	// Structures setting up everything to calculate mutual information (MI)/correlations (Pearson) on input data + multi-dimensional scaling (MDS) + vector quantization (VQ)
	m_layerInput = new PopulationColumns(this,m_nrInputHypercolumns,m_nrInputRateUnits,PopulationColumns::Graded,MPIDistribution::ParallelizationHypercolumns); // input
	AddPopulation(m_layerInput);
	m_layer3 = new PopulationColumns(this,m_nrHypercolumns3,m_nrRateUnits3,PopulationColumns::Graded,MPIDistribution::ParallelizationHypercolumns); 
	PopulationModifier* wta = new WTA();
	m_layer3->AddPopulationModifier(wta);
	FullConnectivity* full10 = new FullConnectivity();
	m_layer3->AddPre(m_layerInput,full10); // Feedforward
	m_bcpnn = new ProjectionModifierBcpnnOnline(0.0001, 10e-6); //real value for 11000: 0.00001
	full10->AddProjectionsEvent(m_bcpnn);
	m_bcpnn->SwitchOnOff(false);
	AddPopulation(m_layer3);
	//m_structureInput->MDSHypercolumns()->SetUseThreshold(true,0.9);
	AddTiming(this);
	AddTiming(m_layerInput);
	AddTiming(m_layer3);
}

void NetworkTemporal3BCPNN_base::NetworkSetupMeters()
{
	m_activityMeter = new Meter("ClassOutput.csv",Storage::CSV);
	m_activityMeter->AttachPopulation(m_layer3);
	//m_activityMeter->SwitchOnOff(false);
	AddMeter(m_activityMeter);
	m_inputMeter = new Meter("inputLayer.csv", Storage::CSV);
	m_inputMeter->AttachPopulation(m_layerInput);
	AddMeter(m_inputMeter);
}

void NetworkTemporal3BCPNN_base::NetworkSetupParameters()
{
	// not used
}


vector<float> NetworkTemporal3BCPNN_base::toBinary(const vector<float>& data, int nrHc, int nrMc)
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

void NetworkTemporal3BCPNN_base::TrainBCPNN(const vector<vector<float> >& trainingData, const vector<vector<float> >& trainingLabels, PopulationColumns* inputLayer, int iterationsBCPNN) {
	vector<float> prevLabels;
	for (int k=0;k<iterationsBCPNN;k++) {
		for(int j=0;j<trainingData.size();j++) {
			if(prevLabels.empty()&&j==0) {
				m_bcpnn->SwitchOnOff(false);
				prevLabels=trainingLabels[j];
				m_layerInput->SetValuesAll(m_dataAssigner->prepareValues(j,trainingData));
				this->Simulate();
				m_bcpnn->SwitchOnOff(true);
			}
			vector<float> v(m_nrRateUnits3,0.0);
			for(int i=0;i<prevLabels.size();++i)
				v[(int)prevLabels[i]]=1.0;
			prevLabels=trainingLabels[j];
			m_layerInput->SetValuesAll(m_dataAssigner->prepareValues(j,trainingData));
			inputLayer->SetValuesAll(v);
			this->Simulate();
			if(m_verbose && this->MPIGetNodeId() == 0)
				cout << j << "(" << k << ") ";
		}
		if(m_verbose && this->MPIGetNodeId() == 0)
				cout <<endl;
	}
	this->Simulate();
	m_bcpnn->SwitchOnOff(false);
}

void NetworkTemporal3BCPNN_base::NetworkRun()
{
	int nrColors = 2;
	char* filenameTrain, *filenameTest, *filenameTrainLabels;
	int nrTrainImages = 1000;
	int nrTestImages = 1000;

	int iterationsPatches = 10;

	if(m_architecture == PC)
	{
		nrTrainImages = 300;
		nrTestImages = 300;
		//iterationsPatches = 10;
		/*filenameTrain="D:\\Databases\\tidigits\\trainData.csv";
		filenameTrainLabels="D:\\Databases\\tidigits\\trainDataLabels.csv";
		filenameTest="D:\\Databases\\tidigits\\trainData.csv";*/
		filenameTrain="D:\\Databases\\timit\\train.csv";
		filenameTrainLabels="D:\\Databases\\timit\\trainL.csv";
		filenameTest="D:\\Databases\\timit\\test.csv";
	}
	else {
		nrTrainImages = 250000;
		nrTestImages = 250000;
		filenameTrain="/cfs/klemming/nobackup/p/paherman/Tin/data/timit/old/train.csv";
		filenameTrainLabels="/cfs/klemming/nobackup/p/paherman/Tin/data/timit/old/trainL.csv";
		filenameTest="/cfs/klemming/nobackup/p/paherman/Tin/data/timit/old/test.csv";
		/*filenameTrain="/cfs/klemming/nobackup/p/paherman/Tin/data/timit/train.csv";
		filenameTrainLabels="/cfs/klemming/nobackup/p/paherman/Tin/data/timit/trainL.csv";
		filenameTest="/cfs/klemming/nobackup/p/paherman/Tin/data/timit/test.csv";*/
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
	// Turn off recording during training
	m_layerInput->SetRecording(false);
	m_layer3->SwitchOnOff(false);
	m_layer3->SetRecording(false);

	int iterationsBCPNN=3;
	m_layerInput->SwitchOnOff(true);
	// turn on all recording
	m_layerInput->SetRecording(true);
	//m_layer2->SetRecording(true);
	PopulationModifier* wta = new WTA();
	m_layerInput->AddPopulationModifier(wta);
	TrainBCPNN(trainingData, trainingLabels, m_layer3, iterationsBCPNN);
	m_layer3->SwitchOnOff(true);
	for(int i=0;i<trainingData.size();i++)
	{
		if(i==1)
			m_layer3->SetRecording(true);
		m_layerInput->SetValuesAll(m_dataAssigner->prepareValues(i,trainingData));
		this->Simulate();
	}
	for(int i=0;i<testData.size();i++)
	{
		m_layerInput->SetValuesAll(m_dataAssigner->prepareValues(i,testData));
		this->Simulate();
	}
	this->Simulate();
	this->Simulate();
	TimingStop("RunThrough");
}