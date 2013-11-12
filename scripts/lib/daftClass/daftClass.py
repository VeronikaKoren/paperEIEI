__all__ = ["PGM", "Node", "Edge", "Plate"]


__version__ = "0.0.3"


import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.patches import FancyArrow
from matplotlib.patches import Rectangle as Rectangle
import numpy as np

class PGM(object):
    """
    The base object for building a graphical model representation.

    :param shape:
        The number of rows and columns in the grid.

    :param origin:
        The coordinates of the bottom left corner of the plot.

    :param grid_unit: (optional)
        The size of the grid spacing measured in centimeters.

    :param node_unit: (optional)
        The base unit for the node size. This is a number in centimeters that
        sets the default diameter of the nodes.

    :param observed_style: (optional)
        How should the "observed" nodes be indicated? This must be one of:
        ``"shaded"``, ``"inner"`` or ``"outer"`` where ``inner`` and
        ``outer`` nodes are shown as double circles with the second circle
        plotted inside or outside of the standard one, respectively.

    :param node_ec: (optional)
        The default edge color for the nodes.

    :param directed: (optional)
        Should the edges be directed by default?

    :param aspect: (optional)
        The default aspect ratio for the nodes.

    """
    def __init__(self, shape, origin=[0, 0],
            grid_unit=2, node_unit=1,
            observed_style="shaded",
            line_width=1, node_ec="k",
            directed=True, aspect=1.0):
        self._nodes = {}
        self._edges = []
        self._plates = []
	self.sum_x=0;
	self.sum_y=0;
	self.center_x=0.;
	self.center_y=0.;
        self._ctx = _rendering_context(shape=shape, origin=origin,
                grid_unit=grid_unit, node_unit=node_unit,
                observed_style=observed_style, line_width=line_width,
                node_ec=node_ec, directed=directed, aspect=aspect)

    def add_node(self, node):
        """
        Add a :class:`Node` to the model.

        :param node:
            The :class:`Node` instance to add.

        """
        self._nodes[node.name] = node
	self.sum_x+=node.x;
	self.sum_y+=node.y;
	self.center_x=self.sum_x/len(self._nodes);
	self.center_y=self.sum_y/len(self._nodes);
        return node

    def add_edge(self, name, name1, name2, directed=None, **kwargs):
        """
        Construct an :class:`Edge` between two named :class:`Node` objects.

        :param name1:
            The name identifying the first node.

        :param name2:
            The name identifying the second node. If the edge is directed,
            the arrow will point to this node.

        :param directed: (optional)
            Should this be a directed edge?

        """
        if directed is None:
            directed = self._ctx.directed

	if self._nodes[name1].isAllBusy==False and self._nodes[name2].isAllBusy==False :
		"""selects free slots for preUnit"""
		freePreSlotsIndx=np.where(self._nodes[name1].isSlotBusy==False)[0];
		
		if len(freePreSlotsIndx)==0:
			self._nodes[name1].isAllBusy==False;
		else:
			nFreePreSlots=len(freePreSlotsIndx);
		
			"""selects free slots for postUnit"""	
			freePostSlotsIndx=np.where(self._nodes[name2].isSlotBusy==False)[0];
			
			if len(freePostSlotsIndx)==0:
				self._nodes[name2].isAllBusy==False;
			else:
				nFreePostSlots=len(freePostSlotsIndx);
		
				"""get their position"""
				preSlots_x=self._nodes[name1].x+self._nodes[name1].x_slot[freePreSlotsIndx];
				preSlots_y=self._nodes[name1].y+self._nodes[name1].y_slot[freePreSlotsIndx];
				postSlots_x=self._nodes[name2].x+self._nodes[name2].x_slot[freePostSlotsIndx];
				postSlots_y=self._nodes[name2].y+self._nodes[name2].y_slot[freePostSlotsIndx];
		
				"""select depending on specific distances"""
				if (self._nodes[name1].name==self._nodes[name2].name):
					#if autapse, select the preSlot that is the most opposite to the center of the graph
					distCenter=np.array([(preSlots_x[i]-self.center_x)**2+(preSlots_y[i]-self.center_y)**2 for i in xrange(nFreePreSlots)]);
					selectedPreSlotIndx=np.where(distCenter==distCenter.max())[0][0];
					#then choose the postSlot the most perpendicular to the preSlot
					distPrePost=np.array([(preSlots_x[selectedPreSlotIndx]-postSlots_x[j])**2+(preSlots_y[selectedPreSlotIndx]-postSlots_y[j])**2 for j in xrange(nFreePostSlots)]);
					maxDistPrePost=distPrePost.max();
					distPrePost[selectedPreSlotIndx]=maxDistPrePost+1.;
					distPrePost=(distPrePost-self._nodes[name1].slotTheta**2)**2;#if post=pre, then takes slots that are distances of two slots 
					selectedPostSlotIndx=np.where(distPrePost==distPrePost.min())[0][0];
				else:
					distPrePost=np.array([[(preSlots_x[i]-postSlots_x[j])**2+(preSlots_y[i]-postSlots_y[j])**2 for i in xrange(nFreePreSlots)] for j in xrange(nFreePostSlots)]);
					selectedSlotsIndx=np.where(distPrePost==distPrePost.min());
					selectedPreSlotIndx=selectedSlotsIndx[1][0];
					selectedPostSlotIndx=selectedSlotsIndx[0][0];
		
				whoPreIndx=freePreSlotsIndx[selectedPreSlotIndx];
				self._nodes[name1].isSlotBusy[whoPreIndx]=True;
				self._nodes[name1].whoIs[0][whoPreIndx]="Pre";
				self._nodes[name1].whoIs[1][whoPreIndx]=self._nodes[name1].name;
		
				whoPostIndx=freePostSlotsIndx[selectedPostSlotIndx];
				self._nodes[name2].isSlotBusy[whoPostIndx]=True;
				self._nodes[name2].whoIs[0][whoPostIndx]="Post";
				self._nodes[name2].whoIs[1][whoPostIndx]=self._nodes[name2].name;
		
				"""build the connection"""
        			e = Edge(name, self._nodes[name1], self._nodes[name2], 
				[self._nodes[name1].x_slot[whoPreIndx],self._nodes[name1].y_slot[whoPreIndx],self._nodes[name2].x_slot[whoPostIndx],self._nodes[name2].y_slot[whoPostIndx]],directed=directed,**kwargs);
        			self._edges.append(e);
				
				#print(self._nodes[name1].name);
				#print(self._nodes[name1].isSlotBusy);
				#print(self._nodes[name1].whoIs);
				#print(self._nodes[name1].x_slot[whoPreIndx]);
				#print(self._nodes[name1].y_slot[whoPreIndx]);
				#print(" ");
				#print(self._nodes[name2].name);
				#print(self._nodes[name2].isSlotBusy);
				#print(self._nodes[name2].whoIs);
				#print(self._nodes[name2].x_slot);
				#print(self._nodes[name2].y_slot);
				#print(self._nodes[name2].x_slot[whoPostIndx]);
				#print(self._nodes[name2].y_slot[whoPostIndx]);
				#print(" ");
				#print(" ");
				#print("dist")
				#print(dist)
				#print(closestPreSlotIndx)
				#print(closestPostSlotIndx)
				
        			return e;
	else:
		print("%s : %r, %s : %r, there is no more space",self._nodes[name1].name,self._nodes[name1].isAllBusy,self._nodes[name2].name,self._nodes[name2].isFull);

    def add_plate(self, plate):
        """
        Add a :class:`Plate` object to the model.

        """
        self._plates.append(plate)
        return None

    def render(self):
        """
        Render the :class:`Plate`, :class:`Edge` and :class:`Node` objects in
        the model. This will create a new figure with the correct dimensions
        and plot the model in this area.

        """
        self.figure = self._ctx.figure()
        self.ax = self._ctx.ax()

        for plate in self._plates:
            plate.render(self._ctx)

        for edge in self._edges:
            edge.render(self._ctx)

        for name, node in self._nodes.iteritems():
            node.render(self._ctx)

        return self.ax


class Node(object):
    """
    The representation of a random variable in a :class:`PGM`.

    :param name:
        The plain-text identifier for the node.

    :param content:
        The display form of the variable.

    :param x:
        The x-coordinate of the node in *model units*.

    :param y:
        The y-coordinate of the node.

    :param scale: (optional)
        The diameter (or height) of the node measured in multiples of
        ``node_unit`` as defined by the :class:`PGM` object.

    :param aspect: (optional)
        The aspect ratio width/height for elliptical nodes; default 1.

    :param observed: (optional)
        Should this be a conditioned variable?

    :param fixed: (optional)
        Should this be a fixed (not permitted to vary) variable?
        If `True`, modifies or over-rides ``diameter``, ``offset``,
        ``facecolor``, and a few other ``plot_params`` settings.
        This setting conflicts with ``observed``.

    :param offset: (optional)
        The ``(dx, dy)`` offset of the label (in points) from the default
        centered position.

    :param plot_params: (optional)
        A dictionary of parameters to pass to the
        :class:`matplotlib.patches.Ellipse` constructor.

    """
    def __init__(self, name, content, x, y, scale=1, aspect=None,
                 observed=False, fixed=False,
                 offset=[0, 0], plot_params={}, nslots=4, slotTheta=2):
        # Node style.
        assert not (observed and fixed), \
                "A node cannot be both 'observed' and 'fixed'."
        self.observed = observed
        self.fixed = fixed

        # Metadata.
        self.name = name
        self.content = content

        # Coordinates and dimensions.
        self.x, self.y = x, y
        self.scale = scale
        self.aspect = aspect

        # Display parameters.
        self.plot_params = dict(plot_params)

        # Text parameters.
        self.offset = list(offset)
        self.va = "center"

        # TODO: Make this depend on the node/grid units.
        if self.fixed:
            self.offset[1] += 6
            self.scale /= 6.
            self.va = "bottom"
            self.plot_params["fc"] = "k"
	
	# Location for edges
	self.nslots=nslots;
	self.slotTheta=slotTheta;
	self.isSlotBusy=np.array([False for i in xrange(nslots)]);
	self.whoIs=np.array([["None" for i in xrange(nslots)] for j in xrange(2)]);
	self.isAllBusy=False;
	self.countBusy=0;
	self.x_slot=np.array([np.cos(float(i)*2.*np.pi/self.nslots) for i in xrange(nslots)]);
	self.y_slot=np.array([np.sin(float(i)*2.*np.pi/self.nslots) for i in xrange(nslots)]);

    def render(self, ctx):
        """
        Render the node.

        :param ctx:
            The :class:`_rendering_context` object.

        """
        # Get the axes and default plotting parameters from the rendering
        # context.
        ax = ctx.ax()
        diameter = ctx.node_unit * self.scale
        if self.aspect is not None:
            aspect = self.aspect
        else:
            aspect = ctx.aspect

        p = dict(self.plot_params)
        p["lw"] = _pop_multiple(p, ctx.line_width, "lw", "linewidth")

        p["ec"] = p["edgecolor"] = _pop_multiple(p, ctx.node_ec,
                                                 "ec", "edgecolor")

        p["fc"] = _pop_multiple(p, "none", "fc", "facecolor")
        fc = p["fc"]

        p["alpha"] = p.get("alpha", 1)

        # Set up an observed node. Note the fc INSANITY.
        if self.observed:
            # Update the plotting parameters depending on the style of
            # observed node.
            d = float(diameter)
            if ctx.observed_style == "shaded":
                p["fc"] = "0.7"
            elif ctx.observed_style == "outer":
                d = 1.1 * diameter
            elif ctx.observed_style == "inner":
                d = 0.9 * diameter
                p["fc"] = fc

            # Draw the background ellipse.
            bg = Ellipse(xy=ctx.convert(self.x, self.y),
                         width=d * aspect, height=d,
                         **p)
            ax.add_artist(bg)

            # Reset the face color.
            p["fc"] = fc

        # Draw the foreground ellipse.
        if ctx.observed_style == "inner" and not self.fixed:
            p["fc"] = "none"
        el = Ellipse(xy=ctx.convert(self.x, self.y),
                     width=diameter * aspect, height=diameter, **p)
        ax.add_artist(el)

        # Reset the face color.
        p["fc"] = fc

        # Annotate the node.
        ax.annotate(self.content, ctx.convert(self.x, self.y),
                xycoords="data", ha="center", va=self.va,
                xytext=self.offset, textcoords="offset points")

        return el


class Edge(object):
    """
    An edge between two :class:`Node` objects.

    :param node1:
        The first :class:`Node`.

    :param node2:
        The second :class:`Node`. The arrow will point towards this node.

    :param directed: (optional)
        Should the edge be directed from ``node1`` to ``node2``? In other
        words: should it have an arrow?

    :param plot_params: (optional)
        A dictionary of parameters to pass to the plotting command when
        rendering.

    """
    def __init__(self, name, node1, node2, coord, directed=True, legend="", plot_params={}):
    	self.name  = name
        self.node1 = node1
        self.node2 = node2
        self.directed = directed
        self.legend = legend
        self.plot_params = dict(plot_params)
	self.slotPre_x=coord[0];
	self.slotPre_y=coord[1];
	self.slotPost_x=coord[2];
	self.slotPost_y=coord[3];

    def render(self, ctx):
        """
        Render the edge in the given axes.

        :param ctx:
            The :class:`_rendering_context` object.

        """
        ax = ctx.ax()

        p = self.plot_params
        p["linewidth"] = _pop_multiple(p, ctx.line_width,
                                        "lw", "linewidth");
					
        if self.directed:
            p["ec"] = _pop_multiple(p, "k", "ec", "edgecolor");
            p["fc"] = _pop_multiple(p, "k", "fc", "facecolor");
            p["head_length"] = p.get("head_length", 0.25);p["head_width"] = p.get("head_width", 0.1);
	     
	    # Scale the coordinates appropriately.
	    xPre, yPre = ctx.convert(self.node1.x, self.node1.y);
	    xPost, yPost = ctx.convert(self.node2.x, self.node2.y);
	    
	    #curve the arrow in the good manner
	    curveParam=p["curve"];
	    if xPre==xPost and yPre==yPost :
	    	curveParam+=p["autocurve"];
		cond=(self.slotPost_x-self.slotPre_x>0. and self.slotPre_y>0 and self.slotPost_y>0);
		cond=cond or (self.slotPost_x-self.slotPre_x<0. and self.slotPre_y<0 and self.slotPost_y<0);
		cond=cond or (self.slotPost_y-self.slotPre_y<0. and self.slotPre_x>0 and self.slotPost_x>0);
		cond=cond or (self.slotPost_y-self.slotPre_y>0. and self.slotPre_x<0 and self.slotPost_x<0);
		if cond : 
			curveParam*=-1.;
	    elif self.slotPre_x*self.slotPre_y>0.:
			curveParam*=-1;
	    
	    ax.annotate("",xy=(xPost+(ctx.node_unit/2.)*self.slotPost_x,yPost+(ctx.node_unit/2.)*self.slotPost_y),
	    xytext=(xPre+(ctx.node_unit/2.)*self.slotPre_x,yPre+(ctx.node_unit/2.)*self.slotPre_y),
	    arrowprops=dict(linewidth=5,color=p["ec"],arrowstyle=p["arrowstyle"],connectionstyle="arc3,rad=%f"%curveParam,mutation_scale=200*p["head_width"]));
	    ax.text((xPre+(ctx.node_unit/2.)*self.slotPre_x+xPost+(ctx.node_unit/2.)*self.slotPost_x)/2.,(yPre+(ctx.node_unit/2.)*self.slotPre_y+yPost+(ctx.node_unit/2.)*self.slotPost_y)/2.,self.legend);
	
        else:
            p["color"] = p.get("color", "k")
            x, y, dx, dy = self._get_coords(ctx)
            line = ax.plot([x, x + dx], [y, y + dy], **p)
            return line



class Plate(object):
    """
    A plate to encapsulate repeated independent processes in the model.

    :param rect:
        The rectangle describing the plate bounds in model coordinates.

    :param label: (optional)
        A string to annotate the plate.

    :param label_offset: (optional)
        The x and y offsets of the label text measured in points.

    :param shift: (optional)
        The vertical "shift" of the plate measured in model units. This will
        move the bottom of the panel by ``shift`` units.

    :param rect_params: (optional)
        A dictionary of parameters to pass to the
        :class:`matplotlib.patches.Rectangle` constructor.

    """
    def __init__(self, rect, label=None, label_offset=[5, 5], shift=0,
            rect_params={}):
        self.rect = rect
        self.label = label
        self.label_offset = label_offset
        self.shift = shift
        self.rect_params = dict(rect_params)

    def render(self, ctx):
        """
        Render the plate in the given axes.

        :param ctx:
            The :class:`_rendering_context` object.

        """
        ax = ctx.ax()

        s = np.array([0, self.shift])
        r = np.atleast_1d(self.rect)
        bl = ctx.convert(*(r[:2] + s))
        tr = ctx.convert(*(r[:2] + r[2:]))
        r = np.concatenate([bl, tr - bl])

        p = self.rect_params
        p["ec"] = _pop_multiple(p, "k", "ec", "edgecolor")
        p["fc"] = _pop_multiple(p, "none", "fc", "facecolor")
        p["lw"] = _pop_multiple(p, ctx.line_width, "lw", "linewidth")

        rect = Rectangle(r[:2], *r[2:], **p)
        ax.add_artist(rect)

        if self.label is not None:
            ax.annotate(self.label, r[:2], xycoords="data",
                    xytext=self.label_offset, textcoords="offset points")

        return rect


class _rendering_context(object):
    """
    :param shape:
        The number of rows and columns in the grid.

    :param origin:
        The coordinates of the bottom left corner of the plot.

    :param grid_unit:
        The size of the grid spacing measured in centimeters.

    :param node_unit:
        The base unit for the node size. This is a number in centimeters that
        sets the default diameter of the nodes.

    :param observed_style:
        How should the "observed" nodes be indicated? This must be one of:
        ``"shaded"``, ``"inner"`` or ``"outer"`` where ``inner`` and
        ``outer`` nodes are shown as double circles with the second circle
        plotted inside or outside of the standard one, respectively.

    :param node_ec:
        The default edge color for the nodes.

    :param directed:
        Should the edges be directed by default?

    :param aspect:
        The default aspect ratio for the nodes.

    """
    def __init__(self, **kwargs):
        # Save the style defaults.
        self.line_width = kwargs.get("line_width", 1.0)

        # Make sure that the observed node style is one that we recognize.
        self.observed_style = kwargs.get("observed_style", "shaded").lower()
        styles = ["shaded", "inner", "outer"]
        assert self.observed_style in styles, \
                "Unrecognized observed node style: {0}\n".format(
                        self.observed_style) \
                + "\tOptions are: {0}".format(", ".join(styles))

        # Set up the figure and grid dimensions.
        self.shape = np.array(kwargs.get("shape", [1, 1]))
        self.origin = np.array(kwargs.get("origin", [0, 0]))
        self.grid_unit = kwargs.get("grid_unit", 2.0)
        self.figsize = self.grid_unit * self.shape / 2.54

        self.node_unit = kwargs.get("node_unit", 1.0)
        self.node_ec = kwargs.get("node_ec", "k")
        self.directed = kwargs.get("directed", True)
        self.aspect = kwargs.get("aspect", 1.0)

        # Initialize the figure to ``None`` to handle caching later.
        self._figure = None
        self._ax = None

    def figure(self):
        if self._figure is not None:
            return self._figure
        self._figure = plt.figure(figsize=self.figsize)
        return self._figure

    def ax(self):
        if self._ax is not None:
            return self._ax

        # Add a new axis object if it doesn't exist.
        self._ax = self.figure().add_axes((0, 0, 1, 1), frameon=False,
                xticks=[], yticks=[])

        # Set the bounds.
        l0 = self.convert(*self.origin)
        l1 = self.convert(*(self.origin + self.shape))
        self._ax.set_xlim(l0[0], l1[0])
        self._ax.set_ylim(l0[1], l1[1])

        return self._ax

    def convert(self, *xy):
        """
        Convert from model coordinates to plot coordinates.

        """
        assert len(xy) == 2
        return self.grid_unit * (np.atleast_1d(xy) - self.origin)


def _pop_multiple(d, default, *args):
    """
    A helper function for dealing with the way that matplotlib annoyingly
    allows multiple keyword arguments. For example, ``edgecolor`` and ``ec``
    are generally equivalent but no exception is thrown if they are both
    used.

    *Note: This function does throw a :class:`ValueError` if more than one
    of the equivalent arguments are provided.*

    :param d:
        A :class:`dict`-like object to "pop" from.

    :param default:
        The default value to return if none of the arguments are provided.

    :param *args:
        The arguments to try to retrieve.

    """
    assert len(args) > 0, "You must provide at least one argument to 'pop'."

    results = []
    for k in args:
        try:
            results.append((k, d.pop(k)))
        except KeyError:
            pass

    if len(results) > 1:
        raise TypeError("The arguments ({0}) are equivalent, you can only "
                .format(", ".join([k for k, v in results]))
                + "provide one of them.")

    if len(results) == 0:
        return default

    return results[0][1]
