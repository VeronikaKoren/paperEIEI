from pylab import *
from os import path
sys.path.append("./predictNet/toolsClass/");
import toolsClass
reload(toolsClass)
sys.path.append("./predictNet/daftClass/");
import daftClass
reload(daftClass)

#################################################
#########CONGRAPHCLASS###########################

class conGraphClass():
	########DEFINE A DEFAULT CONSTRUCTOR#########
	@toolsClass.callClass
	def __init__(self,
				 _daftGraph="None",
				 _name="None",
				 _id="None",
				 _prePop="None",
				 _postPop="None",
				 _lineWidthEdge=2,
				 _arrowHeadLength=0.75,
				 _arrowHeadWidth=0.1,
				 _arrowStyle="simple",
				 _colorCon=["blue","red"],
				 _drawCon=True,
				 _noteCon=False,
				 _curveCon=2.,
				 _autoCurveCon=0.5,
				 _filterNullEdge=True):
		self.name=_name;
		self.id=_id;
		self.daftGraph=_daftGraph;
		self.prePop=_prePop;
		self.postPop=_postPop;
		self.conMat=np.zeros((self.prePop.nUnits,self.postPop.nUnits),dtype=float);
		self.lineWidthEdge=_lineWidthEdge;
		self.arrowHeadLength=_arrowHeadLength;
		self.arrowHeadWidth=_arrowHeadWidth;
		self.arrowStyle=_arrowStyle;
		self.colorCon=_colorCon;
		self.noteCon=_noteCon;
		self.drawCon=_drawCon;
		self.noteCon=_noteCon;
		self.curveCon=_curveCon;
		self.autoCurveCon=_autoCurveCon;
		self.filterNullEdge=_filterNullEdge;
		print("conGraphClass : __init__ : done");
	
	########GET ACCESS TO ATTRIBUTES LIKE A DICTIONNARY/JSON OBJECT#########
	def __getitem__(self, key): return self.__dict__[key];
					  
	def __setitem__(self, key, item):self.__dict__[key]=item;
	
	def printAll(self):
		print(self.__dict__);
	
	def printParam(self,name):
		print("%s :"%name);
		print(self[name]);
					  
	#####DRAW FUNCTION#######
	def draw(self):
		if self.prePop.drawNode==True and self.postPop.drawNode==True:
			if self.drawCon==True:
				for pre in xrange(self.prePop.nUnits):
					for post in xrange(self.postPop.nUnits):
						colorEdge=self.colorCon[((int)(sign(self.conMat[pre,post]))+1)/2];
						if self.noteCon==True:
							_leg="$%s_{%s%s%d%d}$"%(self.nameConMat,self.prePop.name,self.postPop.name,pre+1,post+1);
						else:
							_leg="";
						if self.filterNullEdge==True:
							if self.conMat[pre,post]!=0:
								self.daftGraph.add_edge("%s"%self.name,"%s_%d"%(self.prePop.id,pre),"%s_%d"%(self.postPop.id,post),legend=_leg,plot_params={'linewidth':self.lineWidthEdge,'head_length':self.arrowHeadLength,'arrowstyle':self.arrowStyle,'head_width':self.arrowHeadWidth,'ec':colorEdge,'curve':self.curveCon,'autocurve':self.autoCurveCon});
						else:
							self.daftGraph.add_edge("%s"%self.name,"%s_%d"%(self.prePop.id,pre),"%s_%d"%(self.postPop.id,post),legend=_leg,plot_params={'linewidth':self.lineWidthEdge,'head_length':self.arrowHeadLength,'arrowstyle':self.arrowStyle,'head_width':self.arrowHeadWidth,'ec':colorEdge,'curve':self.curveCon,'autocurve':self.autoCurveCon});
			

#################################################
#########POPGRAPHCLASS###########################
class popGraphClass():
	########DEFINE A DEFAULT CONSTRUCTOR#########
	@toolsClass.callClass
	def __init__(self,
				 _daftGraph="None",
				 _name="None",
				 _id="n",
				 _C_x=0,
				 _C_y=0,
				 _radius=1.,
				 _deltaL=1.,
				 _nUnits=0,
				 _geometryType="inputLayer",
				 _drawNode=True,
				 _noteNode=True,
				 _lineWidthNode=2,
				 _contourColorNode="black",
				 _faceColorNode="white",
				 _doDifferentFaceColor=False,
				 _slotTheta=2,
				 _nSlotsCon=16,
				 _theta=10,
				 _theta0=np.pi,
				 _thetaMax=np.pi,
				 _colorFaceTab=["white"]
				 ):
		self.daftGraph=_daftGraph;
		self.name=_name;
		self.id=_id;
		self.C_x=_C_x;
		self.C_y=_C_y;
		self.radius=_radius;
		self.deltaL=_deltaL;
		self.geometry=[];
		self.nUnits=_nUnits;
		self.geometryType=_geometryType;
		self.drawNode=_drawNode;
		self.noteNode=_noteNode;
		self.lineWidthNode=_lineWidthNode;
		self.contourColorNode=_contourColorNode;
		self.faceColorNode=_faceColorNode;
		self.colorFaceTab=_colorFaceTab;
		self.doDifferentFaceColor=_doDifferentFaceColor;
		self.slotTheta=_slotTheta;
		self.nSlotsCon=_nSlotsCon;
		self.theta=_theta;
		self.theta0=_theta0;
		self.thetaMax=_thetaMax;
		
		print("nodeGraphClass : __init__ : done");
	
	########GET ACCESS TO ATTRIBUTES LIKE A DICTIONNARY/JSON OBJECT#########
	def __getitem__(self, key): return self.__dict__[key]; 
	
	def printAll(self):
		print(self.__dict__);
	
	def printParam(self,name):
		print("%s :"%name);
		print(self[name]);

	####ADD/SET/REMOVE FUNCTIONS#####
	def set_Geometry(self):
		if self.geometryType=="layer":
			self.geometry=[[self.C_x+self.deltaL*float(i) for i in xrange(self.nUnits)],[self.C_y for i in xrange(self.nUnits)]];
		elif self.geometryType=="circle":
			self.geometry=[[self.C_x+self.radius*np.cos((-float(i)*self.thetaMax/(2.*self.theta))+self.theta0) for i in xrange(self.nUnits)],[self.C_y+self.radius*np.sin((-float(i)*self.thetaMax/(2.*self.theta))+self.theta0) for i in xrange(self.nUnits)]];

	####DRAW FUNCTION#####
	def draw(self):
		if self.drawNode==True:
			for unit in xrange(self.nUnits):
				if self.noteNode==True:
					_leg=r"$%s_{%d}$"%(self.name,unit+1);
				else:
					_leg="";
				if self.faceColorNode!="":
					isFilled=True;
				else :
					isFilled=False;
				if self.doDifferentFaceColor==False:
					_colo=self.faceColorNode;
				else:
					_colo=self.colorFaceTab[unit%len(self.colorFaceTab)];
					isFilled=True;
				self.daftGraph.add_node(daftClass.Node("%s_%d"%(self.id,unit),_leg,self.geometry[0][unit],self.geometry[1][unit],plot_params={'linewidth':self.lineWidthNode,'fill':isFilled,'ec':self.contourColorNode,'facecolor':_colo},nslots=self.nSlotsCon,slotTheta=self.slotTheta));
		else:
			print("popGraphClass : draw : self.drawNode is False for %s"%self.name);



#################################################
#########NETGRAPHCLASS###########################
class netGraphClass():
	
	########DEFINE A DEFAULT CONSTRUCTOR#########
	def __init__(self,
				 _daftClass="None",
				 _sizeNode=2,
				 _sizeSpace=2,
				 _sizeFont=2,
				 _autoCurve=0.5):
		
		#######daft###################
		self.daftGraph=_daftClass;
		#######net#####################
		self.sizeNode=_sizeNode;
		self.sizeSpace=_sizeSpace;
		#######neurons#####################
		self.popArray=[];
		self.popDic={};
		########connections##############
		self.conArray=[];
		self.conDic={};
		print("netGraphClass : __init__ : done");
			
	########GET ACCESS TO ATTRIBUTES LIKE A DICTIONNARY/JSON OBJECT#########
	def __getitem__(self, key): return self.__dict__[key]; 
	
	def printAll(self):
		print(self.__dict__);
	
	def printParam(self,name):
		print("%s :"%name);
		print(self[name]);
	
	def find(array,field,id):
		for x in array:
			if x.field == id:
				return x;
				break
		else:
			return None;

	####ADD/SET/REMOVE FUNCTIONS#####
	def addPop(self,*args,**kwargs):
		self.popArray.append(popGraphClass(*args,**kwargs));
	def addCon(self,*args,**kwargs):
		self.conArray.append(conGraphClass(*args,**kwargs));
	def addPopDic(self,*args,**kwargs):
		self.popDic.update({"%d"%len(self.popDic):popGraphClass(*args,**kwargs)});
	def addConDic(self,*args,**kwargs):
		self.conDic.update({"%d"%len(self.conDic):conGraphClass(*args,**kwargs)});

	####DRAW FUNCTIONS#####
	def initDaft(self,_width=5,_height=5):
		self.daftGraph=daftClass.PGM([_width,_height],origin=[-1,1],node_unit=self.sizeNode,grid_unit=self.sizeSpace);

	###PLOT FUNCTION######
	def plotGraph(self):
		figure();
		self.daftGraph.render();

