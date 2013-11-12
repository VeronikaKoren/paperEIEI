from brian import *
from scipy.interpolate import interp1d
import scipy
import math
from scipy.stats import bernoulli,poisson,norm,expon,uniform
from scipy.linalg import cholesky
from scipy.optimize import fsolve
from scipy.optimize import leastsq
import _random

##########################################################################
########################TOOLS FUNCTION####################################


############################MATHEMATICAL AND NUMERIC FUNCTIONS##################
def buildMat(_M=1,_N=1,_pConJ=0.1,_pConR=1,_J=1.,_sdJ=0.,_R=10.,_sdR=0.,_dist="bernouilli",_isSym=True,_isUni=False,_type="all"):
	
	####pick values#######
	minDim=min(_N,_M);
	if _dist=="bernouilli":
		temp=np.triu(bernoulli.rvs(_pConJ,size=(_M,_N)));
		temp[:minDim,:minDim]=temp[:minDim,:minDim]-diag(diagonal(temp))+diag(bernoulli.rvs(_pConR,size=(minDim)));
	if _dist=="poisson":
		temp=np.triu(poisson.rvs(_pConJ,size=(_M,_N)));
		temp[:minDim,:minDim]=temp[:minDim,:minDim]-diag(diagonal(temp))+diag(poisson.rvs(_pConR,size=(minDim)));
	elif _dist=="uniform":
		temp=_J+_sdJ*(0.5-np.triu(uniform.rvs(size=(_M,_N))));
		temp[:minDim,:minDim]=temp[:minDim,:minDim]-diag(diagonal(temp))+diag(_R+_sdR*(0.5-uniform.rvs(size=(minDim))));
	elif _dist=="expon":
		temp=np.triu(expon.rvs(_J,size=(_M,_N)));
		temp[:minDim,:minDim]=temp[:minDim,:minDim]-diag(diagonal(temp))+diag(expon.rvs(_R,size=(minDim)));
	elif _dist=="norm":
		temp=_sdJ*np.triu(norm.rvs(_J,size=(_M,_N)));
		temp[:minDim,:minDim]=temp[:minDim,:minDim]-diag(diagonal(temp))+diag(_sdR*norm.rvs(_R,size=(minDim)));
	
	####symmetrize matrix###
	if _isSym==True:
		if _N==_M:
			temp=_J*(temp.T+temp-2*diag(diagonal(temp)))+_R*diag(diagonal(temp));
		else:
			print("buildMat : N!=M, cannot 'symmetrize' the matrix");

	####unitarize matrix###
	if _isUni==True:
		if _isSym==True:
			print("buildMat : WARNING : isUni=True, the matrix will not be symetric...");
		temp,s,VT=svd(temp);
	
	####render dense or sparse matrix####
	if _type=="all":
		return temp;
	elif _type=="sparse":
		return scipy.sparse.lil_matrix(temp);

def forceSymMat(_mat):
	if (_mat.T==_mat).all():
		print("forceSymMat : matrix is already symetric");
		return _mat;
	else:
		temp=triu(_mat);
		print("forceSymMat : matrix is forced to be symetric");
		return temp.T+temp-diag(diagonal(temp));

def filterMat(_mat=np.zeros((0,0),dtype=float),_thre=0.0000001):
	def test(x):
		if abs(x)<_thre:
			return 0;
		else : 
			return x;
	return array([[test(_mat[i,j]) for j in xrange(shape(_mat)[1])] for i in xrange(shape(_mat)[0])]);

def errorSumFunc(params,_OmegaVec):
	_Omega=reshape(_OmegaVec,((int)(sqrt(shape(_OmegaVec)[0])),(int)(sqrt(shape(_OmegaVec)[0]))));
	_Gamma=array(reshape(params,(len(params)/shape(_Omega)[0],shape(_Omega)[0])));
	return sum((np.dot(_Gamma.T,_Gamma)-_Omega)**2);
def errorVecFunc(params,_OmegaVec):
	_Omega=reshape(_OmegaVec,(sqrt(shape(_OmegaVec)[0]),sqrt(shape(_OmegaVec)[0])));
	_Gamma=array(reshape(params,(len(params)/shape(_Omega)[0],shape(_Omega)[0])));
	return reshape(np.dot(_Gamma.T,_Gamma)-_Omega,(shape(_Omega)[0]*shape(_Omega)[0]));

class matClass():
	
	########################################################################
	########DEFINE A DEFAULT CONSTRUCTOR#########
	def __init__(self,_id=0,_name="None",_method="leastsq",_N=1,_M=1,_pConJ=0.2,_pConR=1,_J=1.,_R=2.,_buildSym=True):
		self.id=_id;
		self.name=_name;
		self.N=_N;
		self.M=_M;
		self.pConJ=_pConJ;
		self.pConR=_pConR;
		self.J=_J;
		self.R=_R;
		self.isSquared=False;
		self.buildSym=_buildSym;
		self.isSymmetric=False;
		self.eigenVal=[];
		self.eigenVec=[];
		self.singularVal=[];
		self.U=[];
		self.VT=[];
		self.isDefinitePos=False;
		self.Gamma=[];
		self.x0=np.zeros((0),dtype=float);
		self.OmegaChol=[];
		self.GammaCut=[];
		self.set_Omega(np.zeros((_N,_M),dtype=float));
		self.set_method(_method);
		self.error=0.;
		
		print("matClass : __init__ : done");
	
	########################################################################	
	########GET ACCESS TO ATTRIBUTES LIKE A DICTIONNARY/JSON OBJECT#########
	def __getitem__(self, key): return self.__dict__[key]; 
	
	def printAll(self):
		print(self.__dict__);
	
	def printParam(self,name):
		print("%s :"%name);
		print(self[name]);
	
	########################################################################
	#######METHODS FOR SETTINGS PARAMETERS####################################
	def set_N(self):
		self.N=shape(self.Omega)[0];
	
	def set_M(self,_M):
		self.M=_M;
	
	def set_method(self,_method):
		self.method=_method;
		if self.method=="leastsq":
			self.errorCompute=lambda _x0,_Omega : sum(errorVecFunc(_x0,reshape(_Omega,(self.N*self.N)))**2);
		else:
			self.errorCompute=errorSumFunc;
	
	def computeError(self):
		self.error=self.errorCompute(self.x0,reshape(self.Omega,(self.N*self.N)));
	
	def set_initialGamma(self,_Gamma0):
		if (shape(_Gamma0)-array((self.M,self.N))==0).all():
			self.x0=reshape(_Gamma0,(self.N*self.M));
			self.Gamma=_Gamma0;
		else:
			print("matClass : set_initialGamma : error, _Omega0 proposed has no good dimensions");
	
	#####CHECK THAT THE MATRIX IS SQUARED#####
	def checkSquared(self):
		if self.N!=shape(self.Omega)[1]:
			print("matClass : checkSquared : Omega is not squared");
			self.isSquared=False;
		else:
			self.isSquared=True;
	
	def squarify(self):
		print("matClass : squarify : ...'Squarification' in process...");
		self.Omega=np.zeros((self.N,self.N),dtype=float);
		self.Omega[0:self.N,0:shape(self.Omega)[1]]=self.Omega;
	
	#####CHECK THAT THE MATRIX IS SYMMETRIC#####
	def checkSymmetric(self):
		if ((self.Omega.T==self.Omega).all()):
			self.isSymmetric=True;
		else:
			print("matClass : checkSymmetric : the matrix Omega is not symmetric");
			self.isSymmetric=False;
	
	#####CHECK THAT THE MATRIX IS DEFINITE POSITIVE#####
	def checkDefinitePos(self):
		if self.isSymmetric==True:
			self.eigenVal=linalg.eigvalsh(self.Omega);
			if (self.eigenVal>0).all():
				print("matClass : checkDefinitePos : Omega is positive definite");
				self.isDefinitePos=True;
			else:
				self.isDefinitePos=False;
				print("matClass : checkDefinitePos : Omega isn't positive definite");
		else:
			self.eigenVal=linalg.eigvals(self.Omega);
			if (self.eigenVal>0).all():
				print("matClass : checkDefinitePos : Omega is positive definite");
				self.isDefinitePos=True;
			else:
				self.isDefinitePos=False;
				print("matClass : checkDefinitePos : Omega isn't positive definite");
	
	#####GET THE CHOLESKY DECOMPOSITION OF OMEGA#####
	def computeOmegaChol(self):
		if self.isDefinitePos==True:
			self.OmegaChol=linalg.cholesky(self.Omega).T;
		else:
			print("matClass : computeGammaChol : Omega has no Cholesky decomposition");
	
	#####GET THE SINGULAR VALUE DECOMPOSITION OF OMEGA#####
	def computeSVD(self):
		self.U,self.singularVal,self.VT=linalg.svd(self.Omega);
	
	#####GET THE SINGULAR VALUE DECOMPOSITION OF OMEGA#####
	def computeGammaCut(self):
		if self.isDefinitePos==True and shape(self.U)[0]>0:
			self.OmegaCut=np.dot((diag(self.singularVal)[:self.M,:])**(0.5),self.VT);
	
	def set_Omega(self,_Omega):
		if (array(shape(_Omega))>0).all():
			self.Omega=_Omega;
			self.set_N();
			self.checkSquared();
			if self.isSquared==False:
				self.squarify();
			self.checkSymmetric();
			#if self.isSymmetric==False:
				#self.Omega=forceSymMat(_Omega);
			self.checkDefinitePos();
			if self.isDefinitePos==True:
				self.computeOmegaChol();
			self.computeSVD();
			self.computeGammaCut();
		else:
			print("matClass : set_Omega : Omega not yet defined");
	
	def addCon(self,_who1=0,_who2=0,_J=1.):
		if self.Omega[_who1,_who2]==0:
			self.Omega[_who1,_who2]=_J;
			if self.isSymmetric==True:
				self.Omega[_who2,_who1]=_J;
				print("matClass : addCon : connections between _who1=%d and _who2=%d were added"%(_who1,_who2));
			else:
				print("matClass : addCon : connection from _who1=%d to _who2=%d was added"%(_who1,_who2));
		else:
			print("matClass : addCon : WARNING, there is already a connection here _who1=%d _who2=%d"%(_who1,_who2));
		self.set_Omega(self.Omega);
	
	def removeCon(self,_who1=0,_who2=0):
		if self.Omega[_who1,_who2]!=0:
			self.Omega[_who1,_who2]=0;
			if self.isSymmetric==True:
				self.Omega[_who2,_who1]=0;
				print("matClass : removeCon : connections between _who1=%d and _who2=%d were removed"%(_who1,_who2));
			else:
				print("matClass : removeCon : connection from _who1=%d to _who2=%d was removed"%(_who1,_who2));
		else:
			print("matClass : removeCon : WARNING, there wasn't a connection here _who1=%d _who2=%d"%(_who1,_who2));
		self.set_Omega(self.Omega);

	def buildOmega(self):
		self.set_Omega(buildMat(_M=self.M,_N=self.N,_pConJ=self.pConJ,_pConR=self.pConR,_J=self.J,_R=self.R,_type="all",_isSym=self.buildSym));
	
	def searchOmegaDefPos(self,_maxCount=100):
		self.set_Omega(buildMat(_M=self.M,_N=self.N,_pConJ=self.pConJ,_pConR=self.pConR,_J=self.J,_R=self.R,_type="all",_isSym=self.buildSym));
		count=0;
		while count<_maxCount and self.isDefinitePos==False:
			count+=1;
			self.set_Omega(buildMat(_M=self.M,_N=self.N,_pConJ=self.pConJ,_pConR=self.pConR,_J=self.J,_R=self.R,_type="all",_isSym=self.buildSym));	
		

	def searchGamma(self):
		
		if len(self.x0)==self.M*self.N:
			self.x0+=0.01*scipy.stats.uniform.rvs(size=self.M*self.N);
			
			############################
			#####OPTIMIZE###############
			
			######DOWNHILL#####
			if self.method=="leastsq":
				self.Gamma=reshape(scipy.optimize.leastsq(errorVecFunc,self.x0,args=(reshape(self.Omega,(self.N*self.N))))[0],(self.M,self.N));
			######POWELL#####
			elif self.method=="powell":
				self.Gamma=reshape(scipy.optimize.fmin_powell(errorSumFunc,self.x0,args=(reshape(self.Omega,(1,self.N*self.N))),disp=False),(self.M,self.N));
			######CG#####
			elif self.method=="cg":
				self.Gamma=reshape(scipy.optimize.fmin_cg(errorSumFunc,self.x0,args=(reshape(self.Omega,(1,self.N*self.N))),disp=False),(self.M,self.N));
			######LEASTSQUARE#####
			elif self.method=="downhill":
				self.Gamma=reshape(scipy.optimize.fmin(errorSumFunc,self.x0,args=(reshape(self.Omega,(1,self.N*self.N))),disp=False),(self.M,self.N));
			
			############################
			#######COMPUTE ERROR########
			self.error=self.errorCompute(reshape(self.Gamma,(self.M*self.N)),reshape(self.Omega,(self.N*self.N)));
			if self.error>=0.001:
				print("buildMatClass : searchGamma : WARNING error=%f is getting too big"%self.error);
		else:
			print("matClass : searchGamma : error, need first to set x0");
	
	def driftGamma(self,maxCount=100):
		count=0;
		self.x0=reshape(self.Gamma,(self.M*self.N));
		while count<maxCount and (abs(self.x0)<0.1).any() and self.error<0.001:
			self.x0+=0.1*scipy.stats.uniform.rvs(size=self.M*self.N);
			self.searchGamma();
			self.x0=reshape(self.Gamma,(self.M*self.N));
			#print(abs(self.x0),self.error,maxCount);
			count+=1;
	
	def reduceGammaLine(self,maxCount=100):
		self.x0=reshape(self.Gamma,(self.M*self.N));
		self.error=self.errorCompute(self.x0,self.Omega);
		who=(self.N-1)*self.M;
		#who=(self.M-1)*(self.N)):(self.M*self.N)
		while who<self.N*self.M and self.error<0.001:
			count=0;
			while count<maxCount and (abs(self.x0)[(self.N-1)*self.M:who]>0.001).any() and self.error<0.001:
				temp=self.x0[(self.N-1)*self.M:who];
				#self.x0+=0.1*scipy.stats.uniform.rvs(size=self.M*self.N);
				self.x0[(self.N-1)*self.M:who]=temp/10;
				#print(abs(self.x0)[((self.M-1)*(self.N)):(self.M*self.N)])
				self.searchGamma();
				self.x0=reshape(self.Gamma,(self.M*self.N));
				self.error=self.errorCompute(self.x0,self.Omega);
				#print(abs(self.x0),self.error,maxCount);
				count+=1;
			print(self.Gamma,count)
			who+=1;
		
		if self.error>=0.001:
			print("matClass : driftGamma : WARNING error=%f is getting too big"%self.error);
	
	def printGamma(self,title):
		print(title);
		if (shape(myFindGammaClass.Gamma)>0).all():
			print("Gamma")
			print(myFindGammaClass.Gamma);
			print("GammaTGamma")
			print(np.dot(myFindGammaClass.Gamma.T,myFindGammaClass.Gamma));
		else:
			print("matClass : printGamma : error...Gamma has no shape");
	
	def plotEigenVal(self):
		bar(array([i for i in xrange(len(self.eigenVal))])+0.5,self.eigenVal[array([i for i in xrange(len(self.eigenVal)-1,-1,-1)])]);


