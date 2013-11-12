from brian import *
from scipy.interpolate import interp1d
import scipy
import math
from scipy.stats import bernoulli,poisson,norm,expon,uniform
from scipy.linalg import cholesky
from scipy.optimize import fsolve
from scipy.optimize import leastsq
import os
from os import path
sys.path.append("./predictNet/toolsClass/");
import toolsClass
reload(toolsClass)
sys.path.append("./predictNet/daftClass/");
import daftClass
reload(daftClass)
sys.path.append("./predictNet/netGraphClass/");
import netGraphClass
reload(netGraphClass)
import _random
import time
rc("font", family="serif", size=12);
rc("text", usetex=True);
rc("./weaklensing.tex");


class IFPopulationClass():
	
	########DEFINE A DEFAULT CONSTRUCTOR#########
	def __init__(self,_id=0,_name="None",_net="None",_decod="None",_N=10):
		self.id=_id;
		self.name=_name;
		self.net=_net;
		self.decod=_decod;
		self.N=_N;
		self.neuGroupBrianObj=[];
		self.isBuilded=False;
		self.muExt0=0.;
		self.isCommanded=False;
		self.V_Thres=array([-50.]);
		self.V_Reset=array([-70.]);
		self.V_Rest=array([-60.]);
		self.sigmaV=1.;
		self.isSigmaThres=False;
		self.sigmaThres=0.;
		self.taum=20.;
		self.alreadySpiked=False;
		self.nSpikesTot=0;
		self.ratePop=0.;
		self.recordBins=5.*ms;
		print("IFPopulationClass : __init__ : done");

	########GET ACCESS TO ATTRIBUTES LIKE A DICTIONNARY/JSON OBJECT#########
	def __getitem__(self, key): return self.__dict__[key]; 
	
	def printAll(self):
		print(self.__dict__);
	
	def printParam(self,name):
		print("%s :"%name);
		print(self[name]);
	
class IFConnectionClass():

	########DEFINE A DEFAULT CONSTRUCTOR#########
	def __init__(self,_net="None",_name="None",_target="v"):
		self.net=_net;
		self.name=_name;
		self.conBrianObj=[[[] for post in xrange(self.net.nPop)] for pre in xrange(self.net.nPop)];
		self.isBuilded=False;
		self.nCon=self.net.nPop**2;
		self.isCon=array([[True for post in xrange(self.net.nPop)] for pre in xrange(self.net.nPop)]);
		self.J=np.zeros((self.net.nPop,self.net.nPop),dtype=float);
		self.R=diagonal(self.J);
		self.target=_target;
		self.tauL=np.zeros((self.net.nPop,self.net.nPop),dtype=float);
		self.pCon=np.zeros((self.net.nPop,self.net.nPop),dtype=float);
		self.fixedPresynapticNeurons=False;
		self.seedNumber=None;
		self.whatStruc=array([["random" for post in xrange(self.net.nPop)] for pre in xrange(self.net.nPop)]);
		self.Omega=[[[] for post in xrange(self.net.nPop)] for pre in xrange(self.net.nPop)];
		print("IFConnectionClass : __init__ : done");
	
	########GET ACCESS TO ATTRIBUTES LIKE A DICTIONNARY/JSON OBJECT#########
	def __getitem__(self, key): return self.__dict__[key]; 
	
	def printAll(self):
		print(self.__dict__);
	
	def printParam(self,name):
		print("%s :"%name);
		print(self[name]);
		
	#####METHODS FOR SETTINGS TOOLS############
	def set_Connexion(self,_pre=0,_post=0,_whatStruc="all-to-all",_args="None"):
		self.whatStruc[_pre,_post]=_whatStruc;
		if self.whatStruc[_pre,_post]=="all-to-all":
			self.J[_pre,_post]=_args[0];
		if self.whatStruc[_pre,_post]=="random":
			self.J[_pre,_post]=_args[0];
			self.R[_pre]=_args[1];
			self.pCon[_pre,_post]=_args[2];
		if self.whatStruc[_pre,_post]=="topo":
			self.Omega[_pre][_post]=_args[0];
		print("IFNetworkClass : set_Connexion : %s connectivity is %s for %d to %d"%(self.name,self.whatStruc[_pre,_post],_post,_pre));

class DecoderClass():
	########DEFINE A DEFAULT CONSTRUCTOR#########
	def __init__(self,_id=0,_net="None",_pop="None",_name="None"):
		self.id=_id;
		self.name=_name;
		self.net=_net;
		self.pop=_pop;
		self.neuGroupBrianObj=[];
		self.isBuilded=False;
		self.Gamma=1.+np.zeros((self.net.M,self.pop.N),dtype=float);
		self.tauD=10.;
		print("DecoderClass : __init__ : done");
	
	########GET ACCESS TO ATTRIBUTES LIKE A DICTIONNARY/JSON OBJECT#########
	def __getitem__(self, key): return self.__dict__[key]; 
	
	def printAll(self):
		print(self.__dict__);
	
	def printParam(self,name):
		print("%s :"%name);
		print(self[name]);

	#####METHODS FOR SETTINGS TOOLS############
	def set_Gamma_diagonal(self):
		self.Gamma=array([[float(i==j) for i in xrange(self.pop.N)] for j in xrange(self.net.M)]);


class IFNetworkClass():
	########DEFINE A DEFAULT CONSTRUCTOR#########
	def __init__(self,_nPop=2,_M=1,_nu=0.,_mu=0.):
		
		###run times properties##########
		self.runTime=200.;
		self.simuClock=0.1;
		self.currClock=10.;
		self.recoClock=0.1;
		self.simulationClock=Clock(dt=self.simuClock*ms,order=0);
		self.currentClock=Clock(dt=self.currClock*ms,order=1);
		self.recordClock=Clock(dt=self.recoClock*ms,order=2);
		self.filterSpikes=False;
		self.prevV=[];
		self.printControl=False;
		
		###network and neurons properties###
		self.M=_M;
		self.nu=_nu;
		self.mu=_mu;
		self.muExt_t=array([linspace(0,com,self.runTime/(self.currClock)) for com in xrange(self.M)]);
		self.nPop=_nPop;
		self.neuPop=[IFPopulationClass(_net=self) for i in xrange(_nPop)];
		self.N_Tot=sum([self.neuPop[i].N for i in xrange(_nPop)]);
		
		###connectivity properties###
		self.fastCon=IFConnectionClass(_net=self,_name="fast",_target="v");
		self.slowCon=IFConnectionClass(_net=self,_name="slow",_target="g");
		
		###decoder properties#####
		self.decod=[DecoderClass(_net=self,_pop=self.neuPop[i]) for i in xrange(_nPop)];
		self.decodCon=IFConnectionClass(_net=self,_name="decoder");
		
		###record properties####
		self.maxVRecord=1;
		self.maxXRecord=1;
		self.whoTraceX=[range(0,min(self.M,self.maxXRecord)) for pop in xrange(self.nPop)];
		self.whoTraceV=[range(0,min(self.neuPop[pop].N,self.maxVRecord)) for pop in xrange(self.nPop)];
	
		###plot Options properties########
		self.colorXTab=[
						["Yellow","Gold","LightGoldenrod","Goldenrod","DarkGoldenrod"],
						["Red","Tomato","Salmon","DarkSalmon","LightSalmon"],
						["Brown","SandyBrown","Sienna","SaddleBrown","Chocolate"],
						["DarkViolet","BlueViolet","MediumPurple","Orchid","VioletRed"],
						["MidnightBlue","CornflowerBlue","DarkSlateBlue","MediumSlateBlue","DodgerBlue"],
						["MediumSeaGreen","LightSeaGreen","DarkGreen","LimeGreen","ForestGreen"]
						];
		self.colorVTab=[["CornflowerBlue","MidnightBlue","DarkSlateBlue","MediumSlateBlue","DodgerBlue","DarkSlateGray"],
						["PaleGreen","MediumSeaGreen","LightSeaGreen","DarkGreen","LimeGreen","ForestGreen"],
						["Yellow","Gold","LightGoldenrod","Goldenrod","DarkGoldenrod","DarkKhaki"],
						["Red","Tomato","Salmon","DarkSalmon","LightSalmon","FireBrick"],
						["Brown","SandyBrown","Sienna","Peru","Chocolate","SaddleBrown"],
						["DarkViolet","BlueViolet","MediumPurple","Orchid","VioletRed","DarkOrchid"]
						];
		self.styleXTab=[[[50,1],[50,10],[5,10,5,10],[8, 4, 2, 4, 2, 4]] for i in xrange(0,shape(self.colorVTab)[0])];
		self.figSize=(35,20);
		self.plotCommand=True;
		self.plotInpCurrent=True;
		self.plotSynCurrent=True;
		self.plotVoltage=True;
		self.plotRaster=True;
		self.plotRead=True;
		self.plotDiffRead=True;
		self.plotRate=True;
		
		###################
		###graph object####
		self.netGraph=netGraphClass.netGraphClass();
		#self.netGraph.addPopDic(_popName="neuron");
		#self.netGraph.addPopDic(_popName="read");
		#self.netGraph.addConDic(_conName="input",_prePop="command",_postPop="neuron");
		#self.netGraph.addConDic(_conName="fast",_prePop="neuron",_postPop="neuron");
		#self.netGraph.addConDic(_conName="slow",_prePop="neuron",_postPop="neuron");
		#self.netGraph.addConDic(_conName="output",_prePop="neuron",_postPop="read");
		print("IFNetworkClass : __init__ : done");
	
	########GET ACCESS TO ATTRIBUTES LIKE A DICTIONNARY/JSON OBJECT#########
	def __getitem__(self, key): return self.__dict__[key]; 
	
	def printAll(self):
		print(self.__dict__);
	
	def printParam(self,name):
		print("%s :"%name);
		print(self[name]);
	
	#######METHODS FOR SETTINGS PARAMETERS################################################
	def set_N(self,_pop,_N):
		self.N_Tot-=self.neuPop[_pop].N;
		self.N_Tot+=_N;
		self.neuPop[_pop].N=_N;
		self.whoTraceV[_pop]=range(0,min(self.neuPop[_pop].N,self.maxVRecord));
		self.whoTraceX[_pop]=range(0,min(self.M,self.maxXRecord));
		#self.Gamma=hstack((self.Gamma,np.zeros((self.net.M,_N))));
		self.decod[_pop].Gamma=1.+np.zeros((self.M,_N),dtype=float);
		print("IFPopulationClass : set N=%d for pop=%s"%(_N,self.neuPop[_pop].name));
	
	def set_simulationClock(self,_simuClock):
		self.simuClock=_simuClock
		self.simulationClock=Clock(dt=self.simuClock*ms,order=0);
	
	def set_currentClock(self,_currClock):
		self.currClock=_currClock
		self.currentClock=Clock(dt=self.currClock*ms,order=1);
	
	def set_recordClock(self,_recoClock):
		self.recoClock=_recoClock
		self.recordClock=Clock(dt=self.recoClock*ms,order=2);
	
	def addType(self,_N=20,_nameType="None"):
		self.nPop+=1;
		self.neuGroup.append(PopulationClass(_N,_nameType));
		self.N_Tot+=_N;
		self.fastCon.J=vstack((hstack((self.fastCon.J,np.zeros((self.nPop-1,1)))),np.zeros((1,self.nPop))));
		np.append(self.fastCon.R,0.);
		self.slowCon.pCon=vstack((hstack((self.slowCon.pCon,np.zeros((self.nPop-1,1)))),np.zeros((1,self.nPop))));
		self.slowCon.J=vstack((hstack((self.slowCon.J,np.zeros((self.nPop-1,1)))),np.zeros((1,self.nPop))));
		np.append(self.fastCon.R,0.);
		self.slowCon.pCon=vstack((hstack((self.slowCon.pCon,np.zeros((self.nPop-1,1)))),np.zeros((1,self.nPop))));
		
		for i in xrange(self.nPop):
			self.neuGroup.isBuilded=False;
		self.fastCon.isBuilded=False;
		self.slowCon.isBuilded=False;
		self.decod.isBuilded=False;
			
	def removeType(self,_type=0):
		self.nPop-=1;
		self.N_Tot-=self.neuPop[_type].N;
		self.neuPop.pop(_type);
		self.fastCon.J=np.delete(np.delete(self.fastCon.J,_type,0),_type,1);
		self.fastCon.R=np.delete(self.fastCon.R,_type,0);
		self.fastCon.pCon=np.delete(np.delete(self.fastCon.pCon,_type,0),_type,1);
		self.slowCon.J=np.delete(np.delete(self.slowCon.J,_type,0),_type,1);
		self.slowCon.R=np.delete(self.slowCon.R,_type,0);
		self.slowCon.pCon=np.delete(np.delete(self.slowCon.pCon,_type,0),_type,1);
		for pop in xrange(self.nPop):
			self.neuPop[pop].isBuilded=False;
			self.decod[pop].isBuilded=False;
		self.fastCon.isBuilded=False;
		self.slowCon.isBuilded=False;

	def set_nPop(self,_nPop):
		temp=self.nPop;
		if _nPop>self.nPop:
			for i in xrange(_nPop-temp): 
				self.addType();
			print("IFNetworkClass : Added %d new Types"%(_nPop-temp));
		elif _nPop<self.nPop:
			for i in xrange(temp-_nPop): 
				self.removeType();
			print("IFNetworkClass : Removed %d new Types"%(temp-_nPop));
		else: 
			print("IFNetworkClass : Nothing to add or remove in set_nPop=%d"%_nPop);

	def set_netGraph(self,width=10.,height=10.):
		self.netGraph=netGraphClass.netGraphClass();
		self.netGraph.initDaft(_width=width,_height=height);
		self.netGraph.addPop(_daftGraph=self.netGraph.daftGraph,_id="command",_name="c",_nUnits=self.M,_colorFaceTab=array(self.colorXTab)[:,0]);
		self.netGraph.addPop(_daftGraph=self.netGraph.daftGraph,_id="neuron",_name="V",_nUnits=self.neuPop[0].N,_colorFaceTab=array(self.colorVTab)[0,:]);
		self.netGraph.addPop(_daftGraph=self.netGraph.daftGraph,_id="read",_name="\hat{x}",_nUnits=self.M,_colorFaceTab=array(self.colorXTab)[:,0]);
		self.netGraph.addCon(_daftGraph=self.netGraph.daftGraph,
							 _prePop=self.netGraph.popArray[toolsClass.findIndxObjFromAttr(self.netGraph.popArray,["id"],["command"])[0]],
							 _postPop=self.netGraph.popArray[toolsClass.findIndxObjFromAttr(self.netGraph.popArray,["id"],["neuron"])[0]],
							 _name="\Gamma^{T}",_id="input");
		self.netGraph.addCon(_daftGraph=self.netGraph.daftGraph,
							 _prePop=self.netGraph.popArray[toolsClass.findIndxObjFromAttr(self.netGraph.popArray,["id"],["neuron"])[0]],
							 _postPop=self.netGraph.popArray[toolsClass.findIndxObjFromAttr(self.netGraph.popArray,["id"],["neuron"])[0]],
							 _name="\Omega^{f}",_id="fast");
		self.netGraph.addCon(_daftGraph=self.netGraph.daftGraph,
							 _prePop=self.netGraph.popArray[toolsClass.findIndxObjFromAttr(self.netGraph.popArray,["id"],["neuron"])[0]],
							 _postPop=self.netGraph.popArray[toolsClass.findIndxObjFromAttr(self.netGraph.popArray,["id"],["neuron"])[0]],
							 _name="\Omega^{s}",_id="slow");
		self.netGraph.addCon(_daftGraph=self.netGraph.daftGraph,
							 _prePop=self.netGraph.popArray[toolsClass.findIndxObjFromAttr(self.netGraph.popArray,["id"],["neuron"])[0]],
							 _postPop=self.netGraph.popArray[toolsClass.findIndxObjFromAttr(self.netGraph.popArray,["id"],["read"])[0]],
							 _name="\Gamma",_id="output");
	
	#######METHODS FOR BUILDING TOOLS################################################
	def checkDelayResetCompatibility(self):
		check=True;
		for pop in xrange(self.nPop):
			if (self.neuPop[pop].V_Thres==self.neuPop[pop].V_Reset).all():
				if self.fastCon.tauL[pop,pop]>0.:
					print("IFNetworkClass : checkDelayResetCompatibility : WARNING, delay is implemented in connexion %s to %s but there is no delay"%(self.neuPop[pop].name,self.neuPop[pop].name));
					check=False;
		return check;
	
	def checkGammaNoiseCompatibility(self):
		check=True;
		if self.filterSpikes==False:
			if self.neuPop[0].N>1:
				for pop in xrange(self.nPop):
					if (self.decod[pop].Gamma==self.decod[pop].Gamma[0]).all():
						if self.neuPop[pop].sigmaV==0.:
							if (self.neuPop[pop].V_Thres==self.neuPop[pop].V_Reset).all():
								print("IFNetworkClass : checkGammaNoiseCompatibility : WARNING, Gamma for %s have all the same value, no noise and no reset will converge to sync solution"%(self.neuPop[pop].name));
							if self.neuPop[pop].sigmaThres==0:
								check=False;
		return check;			
	
	def build_Neu(self):
		#####################################
		#build the network features
		start=time.clock();
		print("IFNetworkClass : START build_Populations");
		for pop in xrange(self.nPop):
			
			#build the brian object###
			eqs_Neu=''' ''';
			
			#declare the possible commands
			if self.neuPop[pop].isCommanded==True:
				eqs_Neu+='''muExt_t : mV
					''';
		
			#declare a possible specific Threshold 
			if len(self.neuPop[pop].V_Thres)>1:	
				
				eqs_Neu+='''T : mV
					''';
			#declare a possible specific Reset 
			if (self.neuPop[pop].V_Reset!=self.neuPop[pop].V_Thres).any():
				eqs_Neu+='''R : mV
					''';
				
			#set the leak voltage equation
			eqs_Neu+='''dv/dt=(-(v-self.neuPop[pop].V_Rest[0]*mV)+self.neuPop[pop].muExt0*mV'''
		
			#add possible noise
			if self.neuPop[pop].sigmaV>0.:
				eqs_Neu+='''+self.neuPop[pop].sigmaV*mV*sqrt(self.neuPop[pop].taum*ms)*xi''';
				
			#add potential slow connection with other pop
			if (self.slowCon.isCon[:,pop]==True).any(): 
				eqs_Neu+='''+self.neuPop[pop].taum*g''';
			
			#add the possible commands
			if self.neuPop[pop].isCommanded==True:
				eqs_Neu+='''+muExt_t''';
			
			#divide by the membrane time constant
			eqs_Neu+=''')/(self.neuPop[pop].taum*ms)''';
			
			#set dimension
			eqs_Neu+=''' : mV'''
						
			#change variable
			eqs_Neu+='''
				''';
			
			#set a slow current
			if (self.slowCon.isCon[:,pop]==True).any(): 
				eqs_Neu+='''dg/dt=-g/(self.decod[pop].tauD*ms) : mV''';
					
			print("IFNetworkClass : %s"%eqs_Neu);
		
				
			#####build neurons with a specific threshold#####
			if len(self.neuPop[pop].V_Thres)>1:
				
				if self.filterSpikes==True:
					#####Build a Thresholdfunction that filters spikes####
					class SingleSpikeThreshold(Threshold):
						def __init__(self, _threshold_value, _alreadySpiked):
							self.threshold_default=_threshold_value;
							self.alreadySpiked=_alreadySpiked;
						
						def __call__(self, P):
							
							if self.alreadySpiked==False:
								spikes = (P.v > P.T).nonzero()[0]
								#Only keep the first (could also be a random one, etc) spike if there is more than one
								if len(spikes)>0:
									who=randint(len(spikes));
									self.alreadySpiked=True;
									spikes=spikes[who:(who+1)];
									P.v[spikes]=P.T[spikes];
								#spikes = spikes[:1]
							else:
								#one neuron has already crossed threshold and wait for an inhibitory reset at this next dt...
								#...So we don't need to count an other new spike for it.
								spikes=array([],dtype=int64);
								self.alreadySpiked=False;
							return spikes		
					
					if (self.neuPop[pop].V_Reset==self.neuPop[pop].V_Thres).all():
						####without an explicit Reset (nu=0, normal cost (nu=0) or linear cost (nu>0))###
						self.neuPop[pop].neuGroupBrianObj=NeuronGroup(self.neuPop[pop].N,eqs_Neu,threshold=SingleSpikeThreshold(-50.*mV,self.neuPop[pop].alreadySpiked),clock=self.simulationClock);
					else:
						####with an explicit Reset (nu>0, quadratic cost (nu>0))###
						self.neuPop[pop].neuGroupBrianObj=NeuronGroup(self.neuPop[pop].N,eqs_Neu,threshold=SingleSpikeThreshold(-50.*mV,self.neuPop[pop].alreadySpiked),reset="v=R",clock=self.simulationClock);
				else:
					if (self.neuPop[pop].V_Reset==self.neuPop[pop].V_Thres).all():
						####without an explicit Reset (nu=0, normal cost (nu=0) or linear cost (nu>0))###
						self.neuPop[pop].neuGroupBrianObj=NeuronGroup(self.neuPop[pop].N,eqs_Neu,threshold="v>=T",clock=self.simulationClock);
					else:
						####with an explicit Reset (nu>0, quadratic cost (nu>0))###
						self.neuPop[pop].neuGroupBrianObj=NeuronGroup(self.neuPop[pop].N,eqs_Neu,threshold="v>=T",reset="v=R",clock=self.simulationClock);		
				
			#####build neurons with the same specific threshold#####
			elif len(self.neuPop[pop].V_Reset)==1:
				#####special case N=1 without reset explicit#####
				if self.neuPop[pop].V_Thres[0]==self.neuPop[pop].V_Reset[0]:
					self.neuPop[pop].neuGroupBrianObj=NeuronGroup(self.neuPop[pop].N,eqs_Neu,threshold=self.neuPop[pop].V_Thres[0]*mV,clock=self.simulationClock);
				#####general case with reset different from the threshold, the same for all#####
				else:
					if self.filterSpikes==True:
						#####Build a Thresholdfunction that filters spikes####
						class SingleSpikeThreshold(Threshold):
							def __init__(self, threshold_value):
								self.threshold_default = threshold_value
							def __call__(self, P):
								spikes = (P.v > P.T).nonzero()[0]
								# Only keep the first (could also be a random one, etc) spike if there is more than one
								if len(spikes)>0:
									who=randint(len(spikes));
									spikes=spikes[who:(who+1)];
								return spikes	
					
						self.neuPop[pop].neuGroupBrianObj=NeuronGroup(self.neuPop[pop].N,eqs_Neu,threshold=SingleSpikeThreshold(self.neuPop[pop].V_Thres[0]*mV),
							reset=self.neuPop[pop].V_Reset[0]*mV,clock=self.simulationClock);
					else:
						self.neuPop[pop].neuGroupBrianObj=NeuronGroup(self.neuPop[pop].N,eqs_Neu,threshold=self.neuPop[pop].V_Thres[0]*mV,
																		  reset=self.neuPop[pop].V_Reset[0]*mV,clock=self.simulationClock);
			
			#initialize randomly the voltages
			self.neuPop[pop].neuGroupBrianObj.v=self.neuPop[pop].V_Rest[0]*mV-5*mV*rand(self.neuPop[pop].N);
			
			#implement the possible commands
			if self.neuPop[pop].isCommanded==True:
				inputMatrix=np.dot(self.decod[pop].Gamma.T,self.muExt_t);
				for neu in xrange(self.neuPop[pop].N):
					self.neuPop[pop].neuGroupBrianObj[neu].muExt_t=self.neuPop[pop].taum*TimedArray((inputMatrix[neu,:])*mV,clock=self.currentClock);
				
			self.neuPop[pop].isBuilded=True;
			print("IFNetworkClass : build_Neu : %s Population builded"%self.neuPop[pop].name);
		print("IFNetworkClass : build_Neu : STOP build_Populations duration=%.0fs"%(time.clock()-start));
	
	def build_Con_NeuNeu(self):
		#####################################
		#build fast inhibitory connections neurons to neurons
		start=time.clock();
		print("IFNetworkClass : build_Con_NeuNeu: START build_Connexions");
		for objCon in [self.fastCon,self.slowCon]:
			for pre in xrange(self.nPop):
				for post in xrange(self.nPop):
					if objCon.isCon[pre,post]==True:
						if objCon.whatStruc[pre,post]=="all-to-all":
							objCon.conBrianObj[pre][post]=Connection(self.neuPop[pre].neuGroupBrianObj,self.neuPop[post].neuGroupBrianObj,state=objCon.target,delay=True,clock=self.simulationClock);
							objCon.conBrianObj[pre][post].connect_full(weight=objCon.J[pre,post]*mV,delay=objCon.tauL[pre,post]*ms);
						elif objCon.whatStruc[pre,post]=="random":
							objCon.conBrianObj[pre][post]=Connection(self.neuPop[pre].neuGroupBrianObj,self.neuPop[post].neuGroupBrianObj,state=objCon.target,delay=objCon.tauL[pre,post]*ms,weight=objCon.J[pre,post]*mV,sparseness=objCon.pCon[pre,post],fixed=objCon.fixedPresynapticNeurons,clock=self.simulationClock,seed=objCon.seedNumber);
						elif objCon.whatStruc[pre,post]=="topo":
							#objCon.conBrianObj[pre][post]=Connection(self.neuPop[pre].neuGroupBrianObj,self.neuPop[post].neuGroupBrianObj,state=objCon.target,clock=self.simulationClock,delay=True);
							#objCon.conBrianObj[pre][post].connect_full(self.neuPop[pre].neuGroupBrianObj,self.neuPop[post].neuGroupBrianObj,weight=lambda i,j:objCon.Omega[i,j]*mV,delay=objCon.tauL*ms)
							objCon.conBrianObj[pre][post]=Connection(self.neuPop[pre].neuGroupBrianObj,self.neuPop[post].neuGroupBrianObj,state=objCon.target,clock=self.simulationClock,delay=objCon.tauL[pre,post]*ms);
							objCon.conBrianObj[pre][post].connect_from_sparse(objCon.Omega[pre][post]*mV,column_access=False);
						print("IFNetworkClass : %s connectivity is %s for Population %s to Population %s"%(objCon.name,objCon.whatStruc[pre,post],self.neuPop[pre].name,self.neuPop[post].name));
					objCon.isBuilded=True
		print("IFNetworkClass : build_Con_NeuNeu : STOP build_Connexions duration=%.0fs"%(time.clock()-start));
	
	def build_Decod(self):
		#####################################
		#build the output readings
		start=time.clock();
		print("IFNetworkClass : build_Decod : START build_Decoders");
		for pop in xrange(self.nPop):
			eqs_Out='''
				dv/dt =-v/(self.decod[pop].tauD*ms) : mV
				'''
			self.decod[pop].neuGroupBrianObj=NeuronGroup(self.M,eqs_Out,threshold=None,clock=self.simulationClock);	
			self.decod[pop].isBuilded=True;
			print("IFNetworkClass : %s Decoder builded"%self.decod[pop].name);
		print("IFNetworkClass : build_Decod : STOP build_Decoders duration=%.0fs"%(time.clock()-start));

	def build_Con_NeuDecod(self):
		#####################################
		#build the output readings connection with neu
		start=time.clock();
		self.decodCon.conBrianObj=[];
		for pop in xrange(self.nPop):
			self.decodCon.conBrianObj.append(Connection(self.neuPop[pop].neuGroupBrianObj,self.decod[pop].neuGroupBrianObj, state='v',weight = self.decod[pop].Gamma.T*mV,clock=self.simulationClock));
			self.decodCon.isBuilded=True;	
				
	def runSimu(self,_A=""):
		
		runOK=array([obj.isBuilded for obj in [self.neuPop[pop] for pop in xrange(self.nPop)]+[self.fastCon]+[self.slowCon]+[self.decod[pop] for pop in xrange(self.nPop)]]);
		if runOK.all():
			
			#####################################
			#print global parameter
			print("##################")
			print("GLOBAL PARAMETERS")
			print("")
			print("runTime=%.0f ms"%self.runTime);
			print("timestep=%.2f ms"%self.simuClock);
			print("linear cost nu : %.0f mV"%(self.nu));
			print("quadratic cost mu : %.0f mV"%(self.mu));
			print("filterSpike is %r"%self.filterSpikes);
			
			#####################################
			#print sensory parameters
			print("")
			print("##################")
			print("SENSORY PARAMETERS")
			print("")
			print("A")
			print(_A)
			print("")
			print("decoder time constant")
			for pop in xrange(self.nPop):
				print("%s tauD=%.0f ms lambdaD=%.2f Hz"%(self.neuPop[pop].name,self.decod[pop].tauD,1000./self.decod[pop].tauD));
			print("")
		
			#####################################
			#print neurons parameter
			Nmax=100;
			print("")
			print("##################")
			print("NEURONS PARAMETERS")
			print("")
			print("membrane time constant")
			for pop in xrange(self.nPop):
				print("%s taum=%.0f ms lambdaV=%.2f Hz"%(self.neuPop[pop].name,self.neuPop[pop].taum,1000./self.neuPop[pop].taum));
			print("")
			print("number of neurons")
			for pop in xrange(self.nPop):
				print("%s N=%d"%(self.neuPop[pop].name,self.neuPop[pop].N));
			print("")
			print("$muExt0$")
			for pop in xrange(self.nPop):
				print("pop : %s"%(self.neuPop[pop].muExt0));
			print("")
			print("$Resting Potential$")
			for pop in xrange(self.nPop):
				print("pop : %s"%(self.neuPop[pop].name));
				print(self.neuPop[pop].V_Rest[0:min(len(self.neuPop[pop].V_Rest),Nmax)]);
			print("")
			print("$Threshold$")
			for pop in xrange(self.nPop):
				print("pop : %s"%(self.neuPop[pop].name));
				print(self.neuPop[pop].V_Thres[0:min(len(self.neuPop[pop].V_Thres),Nmax)]);
			print("")
			print("$Reset$")
			for pop in xrange(self.nPop):
				print("pop : %s"%(self.neuPop[pop].name));
				print(self.neuPop[pop].V_Reset[0:min(len(self.neuPop[pop].V_Reset),Nmax)]);
			
			#####################################
			#print connectivity parameter
			print("")
			print("##################")
			print("CONNECTIVITY PARAMETERS")
			print("")
			print("$\Gamma$")
			for pop in xrange(self.nPop):
				print("pop : %s"%(self.neuPop[pop].name));
				print(self.decod[pop].Gamma[0:min(self.M,50),0:min(self.neuPop[pop].N,Nmax)]);
			
			print("")
			print("")
			print("$\Omega_{fast}$")
			for post in xrange(self.nPop):
				for pre in xrange(self.nPop):
					if shape(self.fastCon.Omega[pre][post])[0]>0:
						print("pre : %s, post : %s"%(self.neuPop[pre].name,self.neuPop[post].name));
						if scipy.sparse.isspmatrix_lil(self.fastCon.Omega[pre][post]):
							print(self.fastCon.Omega[pre][post].todense()[0:min(self.neuPop[pop].N,Nmax),0:min(self.neuPop[pop].N,Nmax)]);
						else:
							print(self.fastCon.Omega[pre][post][0:min(self.neuPop[pop].N,Nmax),0:min(self.neuPop[pop].N,Nmax)]);
			
			print("")
			print("")
			print("$\Omega_{slow}$")
			for post in xrange(self.nPop):
				for pre in xrange(self.nPop):
					if shape(self.slowCon.Omega[pre][post])[0]>0:
						print("pre : %s, post : %s"%(self.neuPop[pre].name,self.neuPop[post].name));
						if scipy.sparse.isspmatrix_lil(self.slowCon.Omega[pre][post]):
							print(self.slowCon.Omega[pre][post].todense()[0:min(self.neuPop[pop].N,Nmax),0:min(self.neuPop[pop].N,Nmax)]);
						else:
							print(self.slowCon.Omega[pre][post][0:min(self.neuPop[pop].N,Nmax),0:min(self.neuPop[pop].N,Nmax)]);
					
			print("")
			print("")
			print("$\Omega_{slow}-\Omega_{fast}$ (if integrator case checking, must be 0<J<1)")
			for post in xrange(self.nPop):
				for pre in xrange(self.nPop):
					if shape(self.slowCon.Omega[pre][post])[0]>0:
						print("pre : %s, post : %s"%(self.neuPop[pre].name,self.neuPop[post].name));
						if scipy.sparse.isspmatrix_lil(self.slowCon.Omega[pre][post]):
							print(self.slowCon.Omega[pre][post].todense()[0:min(self.neuPop[pop].N,Nmax),0:min(self.neuPop[pop].N,Nmax)]+self.fastCon.Omega[pre][post].todense()[0:min(self.neuPop[pop].N,Nmax),0:min(self.neuPop[pop].N,Nmax)]);
						else:
							print(self.slowCon.Omega[pre][post][0:min(self.neuPop[pop].N,Nmax),0:min(self.neuPop[pop].N,Nmax)]+self.fastCon.Omega[pre][post][0:min(self.neuPop[pop].N,Nmax),0:min(self.neuPop[pop].N,Nmax)]);

			
			#####################################
			#build the monitor
			self.currentClock.reinit();
			self.simulationClock.reinit();
			self.recordClock.reinit();
			spikeTimes=[];
			spikeCount=[];
			trace_Rates=[];
			trace_Neu=[];
			trace_MuExt_t=[];
			trace_SlowCurrent=[];
			trace_Out=[];
			for pop in xrange(self.nPop):
				temp=array([self.neuPop[pop].neuGroupBrianObj.v[i] for i in xrange(self.neuPop[pop].N)]);
				self.neuPop[pop].neuGroupBrianObj.reinit();
				
				if len(self.neuPop[pop].V_Thres)>1:
					self.neuPop[pop].neuGroupBrianObj.T=self.neuPop[pop].V_Thres*mV;
				
				if (self.neuPop[pop].V_Reset!=self.neuPop[pop].V_Thres).any():
					self.neuPop[pop].neuGroupBrianObj.R=self.neuPop[pop].V_Reset*mV;
				
				self.neuPop[pop].neuGroupBrianObj.v=temp;
				spikeTimes.append(SpikeMonitor(self.neuPop[pop].neuGroupBrianObj));
				spikeCount.append(SpikeCounter(self.neuPop[pop].neuGroupBrianObj));
				trace_Rates.append(PopulationRateMonitor(self.neuPop[pop].neuGroupBrianObj,bin=self.neuPop[pop].recordBins));
				trace_Neu.append(StateMonitor(self.neuPop[pop].neuGroupBrianObj,'v',record=self.whoTraceV[pop],timestep=1,clock=self.recordClock));
				if self.neuPop[pop].isCommanded==True:
					trace_MuExt_t.append(StateMonitor(self.neuPop[pop].neuGroupBrianObj,'muExt_t',record=self.whoTraceV[pop],timestep=1,clock=self.recordClock));
				else:
					trace_MuExt_t.append("None");
				trace_SlowCurrent.append(StateMonitor(self.neuPop[pop].neuGroupBrianObj,'g',record=self.whoTraceV[pop],timestep=1,clock=self.recordClock));
				trace_Out.append(StateMonitor(self.decod[pop].neuGroupBrianObj,'v',record=self.whoTraceX[pop],clock=self.recordClock));
				
			
			######################################
			#create a Magic Network object
			net=MagicNetwork();
			for pre in xrange(self.nPop):
				net.add(self.neuPop[pre].neuGroupBrianObj);
				net.add(self.decod[pre].neuGroupBrianObj);
				net.add(self.decodCon.conBrianObj[pre]);
				for post in xrange(self.nPop):
					for objCon in [self.fastCon,self.slowCon]:
						if objCon.isCon[pre,post]==True:
							net.add(objCon.conBrianObj[pre][post]);
				net.add(spikeTimes[pre]);
				net.add(trace_Neu[pre]);
				net.add(trace_SlowCurrent[pre]);
				if self.neuPop[pre].isCommanded==True:
					net.add(trace_MuExt_t[pre]);
				net.add(trace_Out[pre]);
				
			######################################
			#network control during simulation
			
			#change a bit the thresholds randomly
			for pop in xrange(self.nPop):
				if len(self.neuPop[pop].V_Thres)>1:
					if self.neuPop[pop].isSigmaThres==True:
						@network_operation(self.simulationClock)
						def randomThres():
							self.neuPop[pop].neuGroupBrianObj.T=(self.neuPop[pop].V_Thres+self.neuPop[pop].sigmaThres*scipy.stats.uniform.rvs(size=len(self.neuPop[pop].V_Thres)))*mV;
						net.add(randomThres);
					
			#filter only one neuron that passes through the threshold
			if self.filterSpikes==True:
				#self.neuPop[pop].prevV=array([self.neuPop[pop].neuGroupBrianObj.v[i] for i in xrange(self.neuPop[pop].N)]);
				@network_operation(Clock(dt=0.05*ms))
				def filterSpikes():
					for pop in xrange(self.nPop):
						whoWouldSpike=where(self.neuPop[pop].neuGroupBrianObj.v>self.neuPop[pop].neuGroupBrianObj.T)[0];
						if len(whoWouldSpike)>1:
							for i in xrange(1,len(whoWouldSpike)):
								self.neuPop[pop].neuGroupBrianObj.v[whoWouldSpike[i]]=self.neuPop[pop].prevV[whoWouldSpike[i]];
						self.neuPop[pop].prevV=array([self.neuPop[pop].neuGroupBrianObj.v[i] for i in xrange(self.neuPop[pop].N)]);
				#net.add(filterSpikes);
			
			#display control during simulation
			if self.printControl==True:
				@network_operation(Clock(dt=0.1*ms))
				def printControl():
					who=0;
					#print(self.simulationClock.t,spikeTimes[0].spikes,self.neuPop[0].neuGroupBrianObj.v[who],self.neuPop[0].neuGroupBrianObj.muExt_t[who],self.neuPop[0].neuGroupBrianObj.g[who]);
					print(self.neuPop[0].T)
				net.add(printControl);
		
			#####################################
			#launch simulation
			start=time.clock();
			print("IFNetworkClass : runSimu : START RUN");
			net.run(self.runTime*ms);
			print("IFNetworkClass : runSimu : STOP RUN duration=%.0fs"%(time.clock()-start));
			
			#####################################
			#print some features of the run
			for pop in xrange(self.nPop):
				self.neuPop[pop].ratePop=spikeTimes[pop].nspikes/(self.runTime*ms);
	
			print("")
			print("")
			print("rate Mean");
			for pop in xrange(self.nPop):
				print("<r_Pop_%s(t)>=%.2f Hz <r_Pop_%s(t)/N>=%.2f Hz"%(self.neuPop[pop].name,self.neuPop[pop].ratePop,self.neuPop[pop].name,self.neuPop[pop].ratePop/self.neuPop[pop].N));
				print("<r_Ind_%s(t)>=%.2f Hz"%(self.neuPop[pop].name,mean(array([spikeCount[pop][i] for i in xrange(self.neuPop[pop].N)])/(self.runTime*ms))));
			######################################
			#return output
			
			return [list(),[spikeTimes,trace_Neu,trace_MuExt_t,trace_SlowCurrent,trace_Rates],[trace_Out]],ms,Hz,mV

		else:
			print("IFNetworkClass : runSimu : Error building Steps are=%s"%runOK);
			return [],ms,Hz,mV;
	
	def plotSimu(self="None",outputIF=[],outputRate=[],timeAx=[0,10],VAx=[-80,-20]):
		figure();plt.subplots(figsize=self.figSize);
		nRows=45;
		nCols=10;
		row=0;
		col=0;
		rowLen=5;
		colLen=5;
		
		if len(outputIF[1])>0:
			
			####DEFINE COLORS#####
			colorV=[[] for pop in xrange(self.nPop)];
			styleX=[[] for pop in xrange(self.nPop)];
			countPop=0;
			for pop in xrange(self.nPop):
				countV=0;
				for i in xrange(self.neuPop[pop].N):
					colorV[pop].append(self.colorVTab[countPop][countV]);
					countV+=1;
					if countV==len(self.colorVTab[countPop]):
						countV=0;
				countX=0;
				for i in xrange(self.M):
					styleX[pop].append(self.styleXTab[countPop][countX]);
					countX+=1;
					if countX==len(self.styleXTab[countPop]):
						countX=0;
				countPop+=1;
				if countPop==shape(self.colorVTab)[0]:
					countPop=0;
			
			if self.plotCommand==True:
				#####PLOT COMMANDS######
				row+=rowLen;ax=plt.subplot2grid((nRows,nCols), (row,col), rowspan=rowLen,colspan=colLen);
				for pop in xrange(self.nPop):
					if self.neuPop[pop].isCommanded==True:
						for i in xrange(len(self.whoTraceX[pop])):
							line,=ax.plot(outputRate[1][pop].times/ms,outputRate[1][pop][i]/mV,'-',color=colorV[pop][0],lw=2);
							line.set_dashes(styleX[pop][i]);
				ax.set_ylabel("$c(t)$",fontsize=25);#ax.set_yticks([-70,-50,-30]);ax.set_yticklabels(["$-70$","$-50$","$-30$"],fontsize=20);
				#ax.set_ylim([-1,5]);
				ax.set_ylim([(-0.001+outputRate[1][0][0].min())/mV,(0.001+outputRate[1][0][0].max())/mV]);
				ax.set_xticks([0,100,200,300,400,500]);ax.set_xticklabels([],fontsize=20);ax.set_xlim([-1.,self.runTime+10.]);
				ax.yaxis.set_ticks_position('left');ax.xaxis.set_ticks_position('bottom');ax.tick_params(length=10, width=5, which='major');ax.tick_params(length=5, width=2, which='minor');
		
			if self.plotInpCurrent==True:
				#####PLOT INPUT CURRENT######
				row+=rowLen;ax=plt.subplot2grid((nRows,nCols), (row,col), rowspan=rowLen,colspan=colLen);
				for pop in xrange(self.nPop):
					if outputIF[1][2][pop]!="None":
						for i in xrange(len(self.whoTraceV[pop])):
							line,=ax.plot(outputIF[1][2][pop].times/ms,outputIF[1][2][pop][i]/mV,'--',lw=3,color=colorV[pop][i]);
				ax.set_ylabel("$\mu_{ext}(t)\ $ \n $(mV)$",fontsize=25);#ax.set_yticks([-70,-50,-30]);ax.set_yticklabels(["$-70$","$-50$","$-30$"],fontsize=20);
				#ax.set_ylim([-1,5]);
				ax.set_ylim([(-0.001+outputIF[1][2][0][0].min())/mV,(0.001+outputIF[1][2][0][0].max())/mV]);
				ax.set_xticks([0,100,200,300,400,500]);ax.set_xticklabels([],fontsize=20);
				#ax.set_xlim([-1.,self.runTime+10.]);
				ax.set_xlim(timeAx);
				ax.yaxis.set_ticks_position('left');ax.xaxis.set_ticks_position('bottom');ax.tick_params(length=10, width=5, which='major');ax.tick_params(length=5, width=2, which='minor');
			
			if self.plotSynCurrent==True:		
				#####PLOT SYNAPTIC CURRENT######
				row+=rowLen;ax=plt.subplot2grid((nRows,nCols), (row,col), rowspan=rowLen,colspan=colLen);
				for pop in xrange(self.nPop):
					for i in xrange(len(self.whoTraceV[pop])):
						line,=ax.plot(outputIF[1][3][pop].times/ms,-(outputIF[1][3][pop][i]/mV),lw=3,color=colorV[pop][i]);
						line,=ax.plot(outputIF[1][2][pop].times/ms,outputIF[1][2][pop][i]/(self.neuPop[pop].taum*mV),'--',lw=3,color=colorV[pop][i]);
				ax.set_ylabel("$\mu_{syn}(t)$",fontsize=25);#ax.set_yticks([-70,-50,-30]);ax.set_yticklabels(["$-70$","$-50$","$-30$"],fontsize=20);#ax.set_ylim([-1,5]);
				ax.set_xticks([0,100,200,300,400,500]);ax.set_xticklabels([],fontsize=20);
				#ax.set_xlim([-1.,self.runTime+10.]);
				ax.set_xlim(timeAx);
				ax.yaxis.set_ticks_position('left');ax.xaxis.set_ticks_position('bottom');ax.tick_params(length=10, width=5, which='major');ax.tick_params(length=5, width=2, which='minor');
		
			if self.plotVoltage==True:
				#####PLOT VOLTAGE######
				row+=rowLen;ax=plt.subplot2grid((nRows,nCols), (row,col), rowspan=rowLen,colspan=colLen);
				for pop in xrange(self.nPop):
					if outputIF[1][2][pop]!="None":
						for i in xrange(len(self.whoTraceV[pop])):
							line,=ax.plot(outputIF[1][2][pop].times/ms,self.neuPop[pop].V_Rest+outputIF[1][2][pop][i]/mV,'--',color=colorV[pop][0],lw=1);
				for pop in xrange(self.nPop): 
					for i in xrange(len(self.whoTraceV[pop])):
						ax.plot(outputIF[1][1][pop].times/ms,outputIF[1][1][pop][i]/mV,lw=3,color=colorV[pop][i]);
						for j in xrange(len(outputIF[1][0][pop][i])):
							ax.plot(array([outputIF[1][0][pop][i][j],outputIF[1][0][pop][i][j]])/ms,array([self.neuPop[pop].V_Reset[min(len(self.neuPop[pop].V_Reset)-1,i)],10.]),'-',color=colorV[pop][i],lw=3);
						ax.plot([0.,self.runTime+10.],[self.neuPop[pop].V_Thres[min(len(self.neuPop[pop].V_Reset)-1,i)],self.neuPop[pop].V_Thres[min(len(self.neuPop[pop].V_Reset)-1,i)]],'--',color=colorV[pop][i],lw=1);
						ax.plot([0.,self.runTime+10.],[self.neuPop[pop].V_Reset[min(len(self.neuPop[pop].V_Reset)-1,i)],self.neuPop[pop].V_Reset[min(len(self.neuPop[pop].V_Reset)-1,i)]],'--',color=colorV[pop][i],lw=1);				
				ax.set_ylabel("$V(t)\ $ \n $(mV)$",fontsize=25);
				if VAx==[-80.,-20.]:
					ax.set_yticks([-70,-50,-30]);
					ax.set_yticklabels(["$-70$","$-50$","$-30$"],fontsize=20);
				ax.set_ylim(VAx);
				ax.set_xticks([0,100,200,300,400,500]);ax.set_xticklabels([],fontsize=20);
				#ax.set_xlim([-1.,self.runTime+10.]);
				ax.set_xlim(timeAx);
				ax.yaxis.set_ticks_position('left');ax.xaxis.set_ticks_position('bottom');ax.tick_params(length=10, width=5, which='major');ax.tick_params(length=5, width=2, which='minor');
		
			if self.plotRaster==True:
				#####PLOT RASTER######
				row+=rowLen;
				ax=plt.subplot2grid((nRows,nCols), (row,col), rowspan=rowLen,colspan=colLen);
				countN=0;
				for pop in xrange(self.nPop):
					MyRasterPlot(outputIF[1][0][pop],colorS=colorV[pop],initial=countN,markersize=15.*min(1,(100./self.N_Tot)));
					countN+=self.neuPop[pop].N;
				ax.set_ylabel("$\# neuron$",fontsize=25);
				ax.set_xlabel("$t\ $ \n $(ms)$",fontsize=25);ax.set_xticks([0,250,500]);ax.set_xticklabels([],fontsize=20);
				ax.set_xticks([0,100,200,300,400,500]);ax.set_xticklabels([],fontsize=20);#ax.set_xlim([-1.,self.runTime+10.]);
				ax.set_xlim(timeAx);
				ax.set_yticks([0,(self.N_Tot/4)-1,(self.N_Tot/2)-1,(3*self.N_Tot/4)-1,self.N_Tot-1]);ax.set_ylim([-0.5,(self.N_Tot-1)+0.5]);
				ax.yaxis.set_ticks_position('left');ax.xaxis.set_ticks_position('bottom');ax.tick_params(length=10, width=5, which='major');ax.tick_params(length=5, width=2, which='minor');
				
		if len(outputIF[2])>0:
			
			if outputRate=="None":
				if self.plotRead==True:
					####PLOT READ OUT######
					row+=rowLen;
					ax=plt.subplot2grid((nRows,nCols), (row,col), rowspan=rowLen,colspan=colLen);
					for pop in xrange(self.nPop):
						for i in xrange(len(self.whoTraceX[pop])):
							line,=ax.plot(outputIF[2][0][pop].times/ms,(1./self.decod[pop].tauD)*outputIF[2][0][pop][i]/mV,'-',color=colorV[pop][0],lw=3);
							line.set_dashes(styleX[pop][i]);
					ax.set_ylabel("$\\hat{x}(t)$",fontsize=25);#ax.set_yticks([-20,-10,0,10,20]);ax.set_yticklabels(["$-20$","$-10$","$0$","$10$","$20$"],fontsize=20);
					#ax.set_ylim([-1.,5.]);
					ax.set_xticks([0,100,200,300,400,500,600,700,800,900,1000]);ax.set_xticklabels(["$0$","$100$","$200$","$300$","$400$","$500$","$600$","$700$","$800$","$900$","$1000$"],fontsize=20);ax.set_xlabel("$t\ (ms)$",fontsize=25);
					#ax.set_xlim([-1.,self.runTime+10.]);
					ax.set_xlim(timeAx);
					ax.yaxis.set_ticks_position('left');ax.xaxis.set_ticks_position('bottom');ax.tick_params(length=10, width=5, which='major');ax.tick_params(length=5, width=2, which='minor');
			else:
				if self.plotRead==True:
					####PLOT READ OUT######
					row+=rowLen;
					ax=plt.subplot2grid((nRows,nCols), (row,col), rowspan=rowLen,colspan=colLen);
					for pop in xrange(self.nPop):
						for i in xrange(len(self.whoTraceX[pop])):
							line,=ax.plot(outputIF[2][0][pop].times/ms,(outputIF[2][0][pop][i])/mV,'-',color=colorV[pop][0],lw=3);
							line.set_dashes(styleX[pop][i]);
							line,=ax.plot(outputRate[0][pop].times/ms,outputRate[0][pop][i]/mV,'-',color=colorV[pop][len(colorV[pop])-1],lw=2);
							line.set_dashes(styleX[pop][i]);
							ax.set_ylabel("$x(t),\\hat{x}(t)\ $",fontsize=25);#ax.set_yticks([-20,-10,0,10,20]);ax.set_yticklabels(["$-20$","$-10$","$0$","$10$","$20$"],fontsize=20);
							#ax.set_ylim([-1.,5.]);
							ax.set_xticks([0,100,200,300,400,500,600,700,800,900,1000]);ax.set_xticklabels([],fontsize=20);#ax.set_xticklabels(["$0$","$100$","$200$","$300$","$400$","$500$","$600$","$700$","$800$","$900$","$1000$"],fontsize=20);#ax.set_xlabel("$t\ (ms)$",fontsize=25);
							#ax.set_xlim([-1.,self.runTime+10.]);
							ax.set_xlim(timeAx);
							ax.yaxis.set_ticks_position('left');ax.xaxis.set_ticks_position('bottom');ax.tick_params(length=10, width=5, which='major');ax.tick_params(length=5, width=2, which='minor');
				
				if self.plotDiffRead==True:
					row+=rowLen;
					ax=plt.subplot2grid((nRows,nCols), (row,col), rowspan=rowLen,colspan=colLen);
					for pop in xrange(self.nPop):
						for i in xrange(len(self.whoTraceX[pop])):
							line,=ax.plot(outputIF[2][0][pop].times/ms,(-outputRate[0][pop][i]+outputIF[2][0][pop][i])/mV,'-',color=colorV[pop][0],lw=3);
							line.set_dashes(styleX[pop][i]);
					ax.plot([0.,self.runTime+10.],[0.,0.],'-',color="red",lw=2);
					ax.set_ylabel("$\\hat{x}(t)-x(t)\ $",fontsize=25);#ax.set_yticks([-20,-10,0,10,20]);ax.set_yticklabels(["$-20$","$-10$","$0$","$10$","$20$"],fontsize=20);
					#ax.set_ylim([-1.,5.]);
					#ax.set_xticks([0,100,200,300,400,500,600,700,800,900,1000]);ax.set_xticklabels(["$0$","$100$","$200$","$300$","$400$","$500$","$600$","$700$","$800$","$900$","$1000$"],fontsize=20);ax.set_xlabel("$t\ (ms)$",fontsize=25);
					ax.set_xticks([0,100,200,300,400,500]);ax.set_xticklabels([],fontsize=20);
					#ax.set_xlim([-1.,self.runTime+10.]);
					ax.set_xlim(timeAx);
					ax.yaxis.set_ticks_position('left');ax.xaxis.set_ticks_position('bottom');ax.tick_params(length=10, width=5, which='major');ax.tick_params(length=5, width=2, which='minor');

			if self.plotRate==True:
				#####PLOT FIRING RATES######
				row+=rowLen;ax=plt.subplot2grid((nRows,nCols), (row,col), rowspan=rowLen,colspan=colLen);
				for pop in xrange(self.nPop):
					line,=ax.plot(outputIF[1][4][pop].times/ms,outputIF[1][4][pop].rate/Hz,lw=3,color=colorV[pop][0]);
				ax.set_ylabel("$r_{pop}(t)\ $ \n $(Hz)$",fontsize=25);#ax.set_yticks([-70,-50,-30]);ax.set_yticklabels(["$-70$","$-50$","$-30$"],fontsize=20);#ax.set_ylim([-1,5]);
				ax.set_xticks([0,100,200,300,400,500,600,700,800,900,1000]);ax.set_xticklabels(["$0$","$100$","$200$","$300$","$400$","$500$","$600$","$700$","$800$","$900$","$1000$"],fontsize=20);ax.set_xlabel("$t\ (ms)$",fontsize=25);
				
				#ax.set_xlim([-1.,self.runTime+10.]);
				ax.set_xlim(timeAx);
				ax.yaxis.set_ticks_position('left');ax.xaxis.set_ticks_position('bottom');ax.tick_params(length=10, width=5, which='major');ax.tick_params(length=5, width=2, which='minor');


##########################PLOT FUNCTIONS#####################################
def plotSpike(spikeTimes,ms,Vr,Vs,colorN,_lw,ax):
	for i in range(0,len(spikeTimes)):
		ax.plot(array([spikeTimes[i],spikeTimes[i]])/ms,array([Vr,Vs]),color=colorN,lw=_lw);

def plotRaster(spikeTimes,nNeuPlot):
	
	raster_plot(spikeTimes);
	ylim(nNeuPlot);

def _Mytake_options(myopts, givenopts):
    """Takes options from one dict into another
		
		Any key defined in myopts and givenopts will be removed
		from givenopts and placed in myopts.
		"""
    for k in myopts.keys():
        if k in givenopts:
            myopts[k] = givenopts.pop(k)


def MyRasterPlot(*monitors, **plotoptions):
    """Raster plot of a :class:`SpikeMonitor`
		
		**Usage**
		
		``raster_plot(monitor,options...)``
        Plots the spike times of the monitor
        on the x-axis, and the neuron number on the y-axis
		``raster_plot(monitor0,monitor1,...,options...)``
        Plots the spike times
        for all the monitors given, with y-axis defined by placing a spike
        from neuron n of m in monitor i at position i+n/m
		``raster_plot(options...)``
        Guesses the monitors to plot automagically
		
		**Options**
		
		Any of PyLab options for the ``plot`` command can be given, as well as:
		
		``showplot=False``
        set to ``True`` to run pylab's ``show()`` function
		``newfigure=False``
        set to ``True`` to create a new figure with pylab's ``figure()`` function
		``xlabel``
        label for the x-axis
		``ylabel``
        label for the y-axis
		``title``
        title for the plot    
		``showgrouplines=False``
        set to ``True`` to show a line between each monitor
		``grouplinecol``
        colour for group lines
		``spacebetweengroups``
        value between 0 and 1 to insert a space between
        each group on the y-axis
		``refresh``
        Specify how often (in simulation time) you would like the plot to
        refresh. Note that this will only work if pylab is in interactive mode,
        to ensure this call the pylab ``ion()`` command.
		``showlast``
        If you are using the ``refresh`` option above, plots are much quicker
        if you specify a fixed time window to display (e.g. the last 100ms).
		``redraw``
        If you are using more than one realtime monitor, only one of them needs
        to issue a redraw command, therefore set this to ``False`` for all but
        one of them.
		
		Note that with some IDEs, interactive plotting will not work with the
		default matplotlib backend, try doing something like this at the
		beginning of your script (before importing brian)::
		
        import matplotlib
        matplotlib.use('WXAgg')
		
		You may need to experiment, try WXAgg, GTKAgg, QTAgg, TkAgg.
		"""
    if len(monitors) == 0:
        (monitors, monitornames) = magic.find_instances(SpikeMonitor)
    if len(monitors):
        # OPTIONS
        # Defaults
        myopts = {"title":"", "xlabel":"Time (ms)", "showplot":False, "showgrouplines":False, \
			"spacebetweengroups":0.0, "grouplinecol":"k", "colorS":["black" for i in xrange(10000)],"initial":0,'newfigure':False,
			'refresh':None, 'showlast':None, 'redraw':True}
        if len(monitors) == 1:
            myopts["ylabel"] = 'Neuron number'
        else:
            myopts["ylabel"] = 'Group number'
        # User options
        _Mytake_options(myopts, plotoptions)
        # PLOTTING ROUTINE
        spacebetween = myopts['spacebetweengroups']
        class SecondTupleArray(object):
            def __init__(self, obj):
                self.obj = obj
            def __getitem__(self, i):
                return float(self.obj[i][1])
            def __len__(self):
                return len(self.obj)
        def get_plot_coords(tmin=None, tmax=None):
            allsn = []
            allst = []
            for i, m in enumerate(monitors):
                mspikes = m.spikes;
                if tmin is not None and tmax is not None:
                    x = SecondTupleArray(mspikes)
                    imin = bisect.bisect_left(x, tmin)
                    imax = bisect.bisect_right(x, tmax)
                    mspikes = mspikes[imin:imax]
                if len(mspikes):
                    sn, st = array(mspikes).T
                else:
                    sn, st = array([]), array([])
                st /= ms
                if len(monitors) == 1:
                    allsn = [sn]
                else:
                    allsn.append(i + ((1. - spacebetween) / float(len(m.source))) * sn)
                allst.append(st)
            sn = hstack(allsn)
            st = hstack(allst)
            if len(monitors) == 1:
                nmax = len(monitors[0].source)
            else:
                nmax = len(monitors)
            return st, sn, nmax
        st, sn, nmax = get_plot_coords();
        if myopts['newfigure']:
            figure()
        if myopts['refresh'] is None:
            for i in xrange(len(st)):
				line, = plot(st[i], sn[i]+myopts['initial'], '.',color=myopts['colorS'][int(sn[i])],**plotoptions)
        else:
            line, = plot([], [], '.', **plotoptions)
        if myopts['refresh'] is not None:
            pylab.axis(ymin=0, ymax=nmax)
            if myopts['showlast'] is not None:
                axis(xmin= -myopts['showlast'] / ms, xmax=0)
        ax = gca()
        if myopts['showgrouplines']:
            for i in range(len(monitors)):
                axhline(i, color=myopts['grouplinecol'])
                axhline(i + (1 - spacebetween), color=myopts['grouplinecol'])
        ylabel(myopts['ylabel'])
        xlabel(myopts['xlabel'])
        title(myopts["title"])
        if myopts["showplot"]:
            pylab.show()
        if myopts['refresh'] is not None:
            @network_operation(clock=EventClock(dt=myopts['refresh']))
            def refresh_raster_plot(clk):
                if matplotlib.is_interactive():
                    if myopts['showlast'] is None:
                        st, sn, nmax = get_plot_coords()
                        line.set_xdata(st)
                        line.set_ydata(sn)
                        ax.set_xlim(0, amax(st))
                    else:
                        st, sn, nmax = get_plot_coords(clk._t - float(myopts['showlast']), clk._t)
                        ax.set_xlim((clk.t - myopts['showlast']) / ms, clk.t / ms)
                        line.set_xdata(array(st))
                        line.set_ydata(sn)
                    if myopts['redraw']:
                        draw()
                        get_current_fig_manager().canvas.flush_events()
            monitors[0].contained_objects.append(refresh_raster_plot)

