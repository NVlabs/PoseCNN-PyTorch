# -----------------------------------------------------------------------------
# Copyright (c) 2009-2016 Nicolas P. Rougier. All rights reserved.
# Distributed under the (new) BSD License.
# -----------------------------------------------------------------------------
import numpy as np
from . import _trackball
import platform
if platform.python_version().startswith('3'):
    from .meshutil import *
else:
    from meshutil import *
class Trackball():
    """
    3D trackball transform

    :param float aspect:
       Indicate what is the aspect ratio of the object displayed. This is
       necessary to convert pixel drag move in oject space coordinates.
       Default is None.

    :param float znear:
       Near clip plane. Default is 2.

    :param float zfar: 
       Distance clip plane. Default is 1000.

    :param float theta:
       Angle (in degrees) around the z axis. Default is 45.

    :param float phi: 
       Angle (in degrees) around the x axis. Default is 45.

    :param float distance:
       Distance from the trackball to the object.  Default is 8.

    :param float zoom:
           Zoom level. Default is 35.

    The trackball transform simulates a virtual trackball (3D) that can rotate
    around the origin using intuitive mouse gestures.

    The transform is connected to the following events:

      * ``on_attach``: Transform initialization
      * ``on_resize``: Tranform update to maintain aspect
      * ``on_mouse_scroll``: Zoom in & out (user action)
      * ``on_mouse_grab``: Drag (user action)

    **Usage example**:

      .. code:: python

         vertex = '''
         attribute vec2 position;
         void main()
         {
             gl_Position = <transform>(vec4(position, 0.0, 1.0));
         } '''

         ...
         window = app.Window(width=800, height=800)
         program = gloo.Program(vertex, fragment, count=4)
         ...
         program['transform'] = Trackball(aspect=1)
         window.attach(program['transform'])
         ...
    """

    aliases = { "view"       : "trackball_view",
                "model"      : "trackball_model",
                "projection" : "trackball_projection" }

    def __init__(self, width, height, cam_pos=[0,0,2.0]):
        """
        Initialize the transform.
        """
        self._aspect = 1
        self._znear = 0.2
        self._zfar = 6.0
        theta = 45
        phi = 45
        distance = np.linalg.norm(cam_pos)
        self._distance = -distance
        self._zoom = 1
        self._width = width
        self._height = height
        self._window_aspect = 1.5

        self._trackball = _trackball.Trackball(45,45)
        aspect = self._window_aspect * self._aspect
        self._projection = perspective(self._zoom, aspect,
                                           self._znear, self._zfar)
        self.property = {}
        self._view = np.eye(4, dtype=np.float32)
        self._view[:3, 3] = -np.array(cam_pos)
        self.property["view"] = self._view
        self.property["model"] = np.eye(4)
        # self.set_distance(self._distance)

    @property
    def distance(self):
        """ Distance from the trackball to the object """

        return self._distance

    # @distance.setter
    # def distance(self, distance):
    #     """ Distance from the trackball to the object """

    #     self._view = np.eye(4, dtype=np.float32)
    #     self._view[2, 3] = -self._distance
    #     self.property["view"] = self._view


    @property
    def theta(self):
        """ Angle (in degrees) around the z axis """

        return self._trackball.theta

    @theta.setter
    def theta(self, theta):
        """ Angle (in degrees) around the z axis """

        self._trackball.theta = theta
        self.property["model"] = self._trackball.model


    @property
    def phi(self):
        """ Angle (in degrees) around the x axis """

        return self._trackball.phi

    @phi.setter
    def phi(self, phi):
        """ Angle (in degrees) around the x axis """

        self._trackball.phi = phi
        self.property["model"] = self._trackball.model


    @property
    def zoom(self):
        """ Zoom level (aperture angle in degrees) """

        return self._zoom


    @phi.setter
    def zoom(self, value):
        """ Zoom level (aperture angle in degrees) """

        aspect = self._window_aspect * self._aspect
        self._zoom = min(max(value, 1.0), 179.0)
        self.property['projection'] = glm.perspective(self._zoom, aspect,
                                             self._znear, self._zfar)

    @property
    def aspect(self):
        """ Projection aspect """

        return self._aspect


    @aspect.setter
    def aspect(self, value):
        """ Projection aspect """

        aspect = self._window_aspect * self._aspect
        self.property['projection'] = glm.perspective(self._zoom, aspect,
                                             self._znear, self._zfar)


    def on_attach(self, program):
        self.property["view"] = self._view
        self.property["model"] = self._trackball.model
        self.property["projection"] = self._projection


    def on_resize(self, width, height):
        self._width  = float(width)
        self._height = float(height)
        self._window_aspect = self._width / self._height
        aspect = self._window_aspect * self._aspect
        self.property['projection'] = perspective(self._zoom, aspect,
                                             self._znear, self._zfar)



    def on_mouse_drag(self, x, y, dx, dy, button=None):
        width = self._width
        height = self._height
        x  = (x*2.0 - width)/width
        dx = (2.*dx)/width
        y  = (height - y*2.0)/height
        dy = -(2.*dy)/height
        self._trackball.drag_to(x,y,dx,dy)
        self.property["model"] = self._trackball.model


    def on_mouse_scroll(self, x, y, dx, dy):
        width = self._width
        height = self._height
        aspect = self._window_aspect * self._aspect
        self._zoom = min(max(self._zoom*(1-dy/100), 1.0), 179.0)
        self.property['projection'] = perspective(self._zoom, aspect,
                                             self._znear, self._zfar)

    def reinit(self, cam_pos):
        self._zoom = 1
        self._window_aspect = 1.5

        self._trackball = _trackball.Trackball(45,45)
        aspect = self._window_aspect * self._aspect
        self._projection = perspective(self._zoom, aspect,
                                           self._znear, self._zfar)
        self.property = {}
        self._view = np.eye(4, dtype=np.float32)
        self._view[:3, 3] = -np.array(cam_pos)
        self.property["view"] = self._view
        self.property["projection"] = np.eye(4)        
        self.property["model"] = np.eye(4)
        self._trackball._model = np.eye(4)
