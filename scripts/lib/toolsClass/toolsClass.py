from pylab import *

#get objects selected by specific values in fields, that belongs to a specific array of objects 
def findIndxObjFromAttr(arrayObj,fieldsSelect,namesSelect):
	where=[];
	for o in xrange(len(arrayObj)):
		obj=arrayObj[o];
		matchGlo=0;
		for i in xrange(len(fieldsSelect)):
			if obj[fieldsSelect[i]]==namesSelect[i]:
				matchGlo+=1;
		if matchGlo==len(fieldsSelect):
			where.append(o);
	return np.array(where);

def findObjFromAttr(arrayObj,fieldsSelect,namesSelect):
	where=[];
	for o in xrange(len(arrayObj)):
		obj=arrayObj[o];
		matchGlo=0;
		for i in xrange(len(fieldsSelect)):
			if obj[fieldsSelect[i]]==namesSelect[i]:
				matchGlo+=1;
		if matchGlo==len(fieldsSelect):
			where.append(o);
	return arrayObj[array(where)];

#set specific attribute in an object that belongs to a specific array of objects 
def setAttrFromObjFromAttr(arrayObj,fieldsSelect,namesSelect,fieldsSet,namesSet):
	for obj in arrayObj:
		matchGlo=0;
		for i in xrange(len(fieldsSelect)):
			if obj[fieldsSelect[i]]!=namesSelect[i]:
				matchGlo+=1;
		if matchGlo==len(fieldsSelect):
			for i in xrange(len(fieldsSet)):
				obj[fieldsSet[i]]=namesSet[i];
			break;


#set specific attribute in an object that belongs to a specific array of objects 
def getAttrFromObjFromAttr(arrayObj,fieldsSelect,namesSelect,fieldsSet):
	for obj in arrayObj:
		matchGlo=0;
		for i in xrange(len(fieldsSelect)):
			if obj[fieldsSelect[i]]==namesSelect[i]:
				matchGlo+=1;
		if matchGlo==len(fieldsSelect):
			out=[];
			for i in xrange(len(fieldsSet)):
				out.append(obj[fieldsSet[i]]);
			return out;
			break;	

#call Class
def callClass(func):
	def inner(*args,**kwargs):
		return func(*args,**kwargs);
	return inner;