from brian import *
from scipy.interpolate import interp1d
import scipy
import math
from scipy.stats import bernoulli,poisson,norm,expon,uniform
from scipy.linalg import cholesky
from scipy.optimize import fsolve
from scipy.optimize import leastsq
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
import time

###################################################
#######SOME MATHEMATICAL TOOLS#####################
def arg(z):
	return 2.*np.arctan(np.imag(z)/(np.sqrt(np.imag(z)**2+np.real(z)**2)+np.real(z)));

def setFreq(fMin,reg,fMax):
	freq=list();
	freq.append(fMin);
	while freq[len(freq)-1]<fMax :
		freq.append(np.exp(np.log(freq[len(freq)-1])+reg));
	return freq;

def plotPhi(X,Y,colo,ax):
	indxBegin=list();
	indxBegin.append(0);
	indxEnd=list();
	for k in range(2,len(Y)):
		if(Y[k-1]*Y[k]<0. and (Y[k]>90. or Y[k]<-90.)):
			indxBegin.append(k);
			indxEnd.append(k-1);
	indxEnd.append(len(Y)-1);
	
	for i in range(0,len(indxBegin)):
		ax.plot(array(X)[range(indxBegin[i],indxEnd[i])],array(Y)[range(indxBegin[i],indxEnd[i])],color=colo,lw=5)

###################################################
#######RATE MODEL CLASS############################
class ratePopulationClass():
	
	########DEFINE A DEFAULT CONSTRUCTOR#########
	def __init__(self,_id=0,_name="None",_net="None",_N=10):
		self.id=_id;
		self.name=_name;
		self.net=_net;
		self.neuGroupBrianObj=[];
		self.isBuilded=False;
		self.sigmaV=1.;
		self.tau=1.;
		print("ratePopulationClass : __init__ : done");

	########GET ACCESS TO ATTRIBUTES LIKE A DICTIONNARY/JSON OBJECT#########
	def __getitem__(self, key): return self.__dict__[key]; 
	
	def printAll(self):
		print(self.__dict__);
	
	def printParam(self,name):
		print("%s :"%name);
		print(self[name]);

class rateConnectionClass():

	########DEFINE A DEFAULT CONSTRUCTOR#########
	def __init__(self,_nUnits,_name):
		self.name=_name;
		self.conBrianObj=[];
		self.isBuilded=False;
		self.N_PrePop=_nUnits;
		self.N_PostPop=_nUnits;
		self.nCon=self.N_PrePop*self.N_PostPop;
		self.isCon=array([[True for post in xrange(self.N_PostPop)] for pre in xrange(self.N_PrePop)]);
		print("rateConnectionClass : __init__ : done");

	########GET ACCESS TO ATTRIBUTES LIKE A DICTIONNARY/JSON OBJECT#########
	def __getitem__(self, key): return self.__dict__[key]; 
	
	def printAll(self):
		print(self.__dict__);
	
	def printParam(self,name):
		print("%s :"%name);
		print(self[name]);
	

class rateNetworkClass():
	
	########DEFINE A DEFAULT CONSTRUCTOR#########
	def __init__(self,_nPop=2,_M=3):
		###time properties##########
		self.runTime=200.;
		self.simuClock=0.1;
		self.currClock=10.;
		self.recoClock=0.1;
		self.simulationClock=Clock(dt=self.simuClock*ms,order=0);
		self.currentClock=Clock(dt=self.currClock*ms,order=1);
		self.recordClock=Clock(dt=self.recoClock*ms,order=2);
		
		###network and neurons properties###
		self.nPop=_nPop;
		self.M=_M;
		self.ratePop=[ratePopulationClass() for i in xrange(_nPop)];
		self.A=np.zeros((self.M,self.M),dtype=float);
		self.con=[rateConnectionClass(self.M,"direct") for i in xrange(_nPop)];
		self.muExt0=15.;
		self.muExt_t=[linspace(0,com,self.runTime/(self.currClock)) for com in xrange(self.M)];
		self.f=setFreq(1.,0.01,1000.);
		
		###record properties####
		self.maxXRecord=5;
		self.whoTraceX=[range(0,min(self.M,self.maxXRecord)) for pop in xrange(self.nPop)];
		
		###plot Options properties########
		self.colorXTab=[
						["Yellow","Gold","LightGoldenrod","Goldenrod","DarkGoldenrod"],
						["Red","Tomato","Salmon","DarkSalmon","LightSalmon"],
						["Brown","SandyBrown","Sienna","SaddleBrown","Chocolate"],
						["DarkViolet","BlueViolet","MediumPurple","Orchid","VioletRed"],
						["MidnightBlue","CornflowerBlue","DarkSlateBlue","MediumSlateBlue","DodgerBlue"],
						["MediumSeaGreen","LightSeaGreen","DarkGreen","LimeGreen","ForestGreen"]
						];
		self.styleXTab=[[[50,1],[50,10],[5,10,5,10],[8, 4, 2, 4, 2, 4]] for i in xrange(0,shape(self.colorXTab)[0])];
		print("rateNetworkClass : __init : done");
	
		###################
		###graph object####
		self.netGraph=netGraphClass.netGraphClass();
	
	########GET ACCESS TO ATTRIBUTES LIKE A DICTIONNARY/JSON OBJECT#########
	def __getitem__(self, key): return self.__dict__[key]; 
	
	def printAll(self):
		print(self.__dict__);
	
	def printParam(self,name):
		print("%s :"%name);
		print(self[name]);
	
	#######METHODS FOR BUILDING TOOLS################################################
	def set_netGraph(self,width=10.,height=10.):
		self.netGraph=netGraphClass.netGraphClass();
		self.netGraph.initDaft(_width=width,_height=height);
		self.netGraph.addPop(_daftGraph=self.netGraph.daftGraph,_id="command",_name="c",_nUnits=self.M,_colorFaceTab=array(self.colorXTab)[:,0]);
		self.netGraph.addPop(_daftGraph=self.netGraph.daftGraph,_id="sensory",_name="x",_nUnits=self.M,_colorFaceTab=array(self.colorXTab)[:,0]);
		self.netGraph.addCon(_daftGraph=self.netGraph.daftGraph,
							 _prePop=self.netGraph.popArray[toolsClass.findIndxObjFromAttr(self.netGraph.popArray,["id"],["command"])[0]],
							 _postPop=self.netGraph.popArray[toolsClass.findIndxObjFromAttr(self.netGraph.popArray,["id"],["sensory"])[0]],
							 _name="",_id="input");
		self.netGraph.addCon(_daftGraph=self.netGraph.daftGraph,
							 _prePop=self.netGraph.popArray[toolsClass.findIndxObjFromAttr(self.netGraph.popArray,["id"],["sensory"])[0]],
							 _postPop=self.netGraph.popArray[toolsClass.findIndxObjFromAttr(self.netGraph.popArray,["id"],["sensory"])[0]],
							 _name="A",_id="recurrent");
	
	def set_simulationClock(self,_simuClock):
		self.simuClock=_simuClock
		self.simulationClock=Clock(dt=self.simuClock*ms,order=0);
	
	def set_currentClock(self,_currClock):
		self.currClock=_currClock
		self.currentClock=Clock(dt=self.currClock*ms,order=1);
	
	def set_recordClock(self,_recoClock):
		self.recoClock=_recoClock
		self.recordClock=Clock(dt=self.recoClock*ms,order=2);
	
	def computeTransferFunction(self,c1):												
		return array([np.dot(inv((1.+1j*(1./(2.*pi))*array(self.f[fInd]))*diag(array([1. for i in xrange(self.nPop)]))-self.A),c1) for fInd in xrange(0,len(self.f))])
	
	def build_Neu(self):
		#####################################
		#build the network features
		start=time.clock();
		print("rateModelClass : START build_Populations");
		for pop in xrange(self.nPop):
			eqs = '''muExt_t : mV
				Av : mV
				dv/dt=(muExt_t+Av+self.ratePop[pop].sigmaV*mV*sqrt(self.ratePop[pop].tau*ms)*xi)/(self.ratePop[pop].tau*ms):mV''';
			
			def myreset(P,spikes):
				for i in xrange(len(P.v[spikes])):
					P.v[spikes][i]=0.;
			print("rateNetworkClass : %s"%eqs);
			self.ratePop[pop].neuGroupBrianObj=NeuronGroup(self.M, eqs,threshold=lambda v :(v<=0.),
														   reset=myreset,clock=self.simulationClock);
			for com in xrange(self.M):
				self.ratePop[pop].neuGroupBrianObj.subgroup(1).muExt_t=TimedArray(self.muExt_t[com]*mV,clock=self.currentClock);
			self.ratePop[pop].isBuilded=True;
			print("rateModelClass : build_Neu : %s Population builded"%self.ratePop[pop].name);
		print("rateModelClass : build_Neu : STOP build_Populations duration=%.0fs"%(time.clock()-start));


	def build_Con(self):
		#####################################
		#build the output readings connection with neu
		start=time.clock();
		print("rateModelClass : build_Con : START build_Connexions");
		for pop in xrange(self.nPop):
			self.con[pop].conBrianObj=Synapses(self.ratePop[pop].neuGroupBrianObj,self.ratePop[pop].neuGroupBrianObj,model='''J : 1
				w : mV
				w = J*v_pre : mV''',clock=self.simulationClock);
			self.con[pop].isBuilded=True;
		print("rateModelClass : build_Con : STOP build_Connexions duration=%.0fs"%(time.clock()-start));

	def runSimu(self):
		
		runOK=array([obj.isBuilded for obj in [self.ratePop[pop] for pop in xrange(self.nPop)]+[self.con[pop] for pop in xrange(self.nPop)]]);
		if runOK.all():
			
			#####################################
			#build the monitor
			self.currentClock.reinit();
			self.simulationClock.reinit();
			self.recordClock.reinit();
		
			trace_X=[];
			trace_muExt_t=[];
			for pop in xrange(self.nPop):
				self.ratePop[pop].neuGroupBrianObj.reinit();
				self.con[pop].conBrianObj[:,:]=True;
				self.con[pop].conBrianObj.J[:,:]=reshape(self.A,(1,shape(self.A)[0]*shape(self.A)[1]))[0];
				self.ratePop[pop].neuGroupBrianObj.Av=self.con[pop].conBrianObj.w;
				trace_X.append(StateMonitor(self.ratePop[pop].neuGroupBrianObj,'v',record=self.whoTraceX[pop],timestep=1,clock=self.recordClock));
				trace_muExt_t.append(StateMonitor(self.ratePop[pop].neuGroupBrianObj,'muExt_t',record=self.whoTraceX[pop],timestep=1,clock=self.recordClock));
			
			######################################
			#launch simulation
			net=MagicNetwork();
			for pre in xrange(self.nPop):
				net.add(self.ratePop[pre].neuGroupBrianObj);
				net.add(self.con[pre].conBrianObj[pre]);
				net.add(trace_X[pre]);
				net.add(trace_muExt_t[pre]);
			
			start=time.clock();
			print("rateModelClass : runSimu : START RUN");
			net.run(self.runTime*ms);
			print("rateModelClass : runSimu : STOP RUN duration=%.0fs"%(time.clock()-start));
			
			######################################
			#return output
			return [trace_X,trace_muExt_t],ms,Hz,mV

		else:
			print("rateModelClass : runSimu : Error building Steps are=%s"%runOK);
			return [],ms,Hz,mV;
	
	def plotSimu(self,output):
		figure();plt.subplots(figsize=(30,15));
		nRows=30;
		nCols=10;
		row=0;
		col=0;
		rowLen=5;
		colLen=5;
		
		if len(output[0])>0:
			
			####DEFINE COLORS#####
			styleX=[[] for pop in xrange(self.nPop)];
			countPop=0;
			for pop in xrange(self.nPop):
				countX=0;
				for i in xrange(self.M):
					styleX[pop].append(self.styleXTab[countPop][countX]);
					countX+=1;
					if countX==len(self.styleXTab[countPop]):
						countX=0;
				countPop+=1;
				if countPop==shape(self.styleXTab)[0]:
					countPop=0;
				
			#####PLOT CURRENT######
			row+=rowLen;ax=plt.subplot2grid((nRows,nCols), (row,col), rowspan=rowLen,colspan=colLen);
			for pop in xrange(self.nPop):
				for i in xrange(len(self.whoTraceX[pop])):
					line,=ax.plot(output[1][pop].times/ms,output[1][pop][i]/mV,'-',color=self.colorXTab[pop][0],lw=2);
					line.set_dashes(styleX[pop][i]);
			ax.set_ylabel("$c(t)\ (mV)$",fontsize=25);#ax.set_yticks([-70,-50,-30]);ax.set_yticklabels(["$-70$","$-50$","$-30$"],fontsize=20);
			#ax.set_ylim([-1,5]);
			ax.set_xticks([0,100,200,300,400,500]);ax.set_xticklabels([],fontsize=20);ax.set_xlim([-1.,self.runTime+10.]);
			ax.yaxis.set_ticks_position('left');ax.xaxis.set_ticks_position('bottom');ax.tick_params(length=10, width=5, which='major');ax.tick_params(length=5, width=2, which='minor');
			
			#####PLOT OUTPUT######
			row+=rowLen;ax=plt.subplot2grid((nRows,nCols), (row,col), rowspan=rowLen,colspan=colLen);
			for pop in xrange(self.nPop):
				for i in xrange(len(self.whoTraceX[pop])):
					line,=ax.plot(output[0][pop].times/ms,output[0][pop][i]/mV,'-',color=self.colorXTab[pop][0],lw=2);
					line.set_dashes(styleX[pop][i]);
			ax.set_ylabel("$x(t)\ (mV)$",fontsize=25);#ax.set_yticks([-70,-50,-30]);ax.set_yticklabels(["$-70$","$-50$","$-30$"],fontsize=20);ax.set_ylim([-80,-20]);
			#ax.set_ylim([-1,5]);
			ax.set_xticks([0,100,200,300,400,500]);ax.set_xticklabels([],fontsize=20);ax.set_xlim([-1.,self.runTime+10.]);
			ax.yaxis.set_ticks_position('left');ax.xaxis.set_ticks_position('bottom');ax.tick_params(length=10, width=5, which='major');ax.tick_params(length=5, width=2, which='minor');

	def plotTransfFunc(self,x1):
		figure()
		fontSize=25;
		plt.subplots(figsize=(5,5));
		axAmp=plt.subplot2grid((4,2), (0,0), rowspan=2,colspan=2);
		axPhase=plt.subplot2grid((4,2), (2,0), rowspan=2,colspan=2);
		for pop in xrange(self.nPop):
			axAmp.plot(self.f,abs(x1[:,pop]),color=self.colorXTab[pop][0],lw=5);
			axAmp.set_xscale('log');
			axAmp.set_xticklabels([]);
			axAmp.set_yticks([0.,0.5,1.,1.5]);
			axAmp.set_yticklabels(["$0$","$0.5$","$1$","$1.5$"]);
			axAmp.set_ylabel("$|x_{1}|$");
			for item in ([axAmp.title, axAmp.xaxis.label, axAmp.yaxis.label]):
				item.set_fontsize(fontSize);
			axAmp.set_ylabel('$|r_{1a}|/r_{0a}$');
			for item in (axAmp.get_xticklabels() + axAmp.get_yticklabels()):
				item.set_fontsize(fontSize-5);
			axAmp.yaxis.set_ticks_position('left')
			axAmp.xaxis.set_ticks_position('bottom')
			axAmp.tick_params(length=10, width=5, which='major');
			axAmp.tick_params(length=5, width=2, which='minor')
			plotPhi(self.f,(180./(np.pi))*arg(x1[:,pop]/x1[:,0]),self.colorXTab[pop][0],axPhase);
			axPhase.set_xscale('log');
			axPhase.set_xlabel("$f=\omega/2\pi\ (Hz)$");
			axPhase.set_ylabel("$\phi(x_{1})$");
			axPhase.set_yticks([-180.,-90.,0.,90.,180.]);
			axPhase.set_yticklabels(["$-180$","$-90$","$0$","$90$","$180$"]);
			axPhase.set_ylim([-185.,185.]);
			for item in (axPhase.get_xticklabels() + axPhase.get_yticklabels()):
				item.set_fontsize(fontSize-5);
			axPhase.yaxis.set_ticks_position('left')
			axPhase.xaxis.set_ticks_position('bottom')
			axPhase.tick_params(length=10, width=5, which='major');
			axPhase.tick_params(length=5, width=2, which='minor');

			