# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial. Full
# text can be found in LICENSE.md

import sys
import ctypes
import torch
import time
import argparse
from pprint import pprint
from PIL import Image
import glutils.glcontext as glcontext
import OpenGL.GL as GL
import cv2
import numpy as np
import platform
PYTHON2 = True
if platform.python_version().startswith('3'):
    PYTHON2 = False

from pyassimp import *
from glutils.meshutil import perspective, lookat, xyz2mat, quat2rotmat, mat2xyz, safemat2quat,homotrans, mat2rotmat, unpack_pose, pack_pose
from glutils.trackball import Trackball
from transforms3d.quaternions import axangle2quat, mat2quat, qmult, qinverse
from transforms3d.euler import quat2euler, mat2euler, euler2quat
import CppYCBRenderer
from numpy.linalg import inv, norm
try:
    from .get_available_devices import *
except:
    from get_available_devices import *

MAX_NUM_OBJECTS = 3
from glutils.utils import colormap


def loadTexture(path):
    img = Image.open(path).transpose(Image.FLIP_TOP_BOTTOM)
    img_data = np.fromstring(img.tobytes(), np.uint8)
    width, height = img.size
 
    texture = GL.glGenTextures(1)
    GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
    GL.glBindTexture(GL.GL_TEXTURE_2D, texture)
    GL.glTexParameterf(
        GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
    GL.glTexParameterf(
        GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR_MIPMAP_LINEAR)
    GL.glTexParameterf(
        GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_REPEAT) #.GL_CLAMP_TO_EDGE GL_REPEAT
    GL.glTexParameterf(
        GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_REPEAT)
    if img.mode == 'RGBA':
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA, width, height, 0,
                    GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, img_data)
    else:
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGB, width, height, 0,
                    GL.GL_RGB, GL.GL_UNSIGNED_BYTE, img_data)
    GL.glGenerateMipmap(GL.GL_TEXTURE_2D)
    return texture


class YCBRenderer:
    def __init__(self, width=512, height=512, gpu_id=0, render_marker=False, robot=''):
        self.render_marker = render_marker
        self.VAOs = []
        self.VBOs = []
        self.materials = []
        self.textures = []
        self.is_textured = []
        self.is_materialed = []
        self.objects = []
        self.texUnitUniform = None
        self.width = width
        self.height = height
        self.vertices = []
        self.faces = []
        self.poses_trans = []
        self.poses_rot = []
        self.instances = []
        self.extents = []
        self.robot = robot
        if len(self.robot) > 3:
            self._offset_map = self.load_offset()
        else:
            self._offset_map = None

        self.r = CppYCBRenderer.CppYCBRenderer(width, height, get_available_devices()[gpu_id])
        self.r.init()
        self.glstring = GL.glGetString(GL.GL_VERSION)
        from OpenGL.GL import shaders
        
        self.shaders = shaders
        self.colors = [[0.9, 0, 0], [0.6, 0, 0], [0.3, 0, 0], [0.3, 0, 0], [0.3, 0, 0], [0.3, 0, 0], [0.3, 0, 0]]
        self.lightcolor = [1, 1, 1]

        cur_dir = os.path.dirname(os.path.abspath(__file__))
        vertexShader = self.shaders.compileShader(
                            open(os.path.join(cur_dir, 'shaders/vert.shader')).readlines(), GL.GL_VERTEX_SHADER)

        fragmentShader = self.shaders.compileShader(
                            open(os.path.join(cur_dir,'shaders/frag.shader')).readlines(), GL.GL_FRAGMENT_SHADER)

        vertexShader_textureMat = self.shaders.compileShader(
                            open(os.path.join(cur_dir,'shaders/vert_blinnphong.shader')).readlines(), GL.GL_VERTEX_SHADER)

        fragmentShader_textureMat = self.shaders.compileShader(
                            open(os.path.join(cur_dir,'shaders/frag_blinnphong.shader')).readlines(), GL.GL_FRAGMENT_SHADER)

        vertexShader_textureless = self.shaders.compileShader(
                            open(os.path.join(cur_dir,'shaders/vert_textureless.shader')).readlines(), GL.GL_VERTEX_SHADER)

        fragmentShader_textureless = self.shaders.compileShader(
                            open(os.path.join(cur_dir,'shaders/frag_textureless.shader')).readlines(), GL.GL_FRAGMENT_SHADER)

        #try with the easiest shader first, and then look at Gl apply material
        vertexShader_material = self.shaders.compileShader(
                            open(os.path.join(cur_dir,'shaders/vert_mat.shader')).readlines(), GL.GL_VERTEX_SHADER)

        fragmentShader_material = self.shaders.compileShader(
                            open(os.path.join(cur_dir,'shaders/frag_mat.shader')).readlines(), GL.GL_FRAGMENT_SHADER)

        vertexShader_simple = self.shaders.compileShader(
                            open(os.path.join(cur_dir,'shaders/vert_simple.shader')).readlines(), GL.GL_VERTEX_SHADER)

        fragmentShader_simple = self.shaders.compileShader(
                            open(os.path.join(cur_dir,'shaders/frag_simple.shader')).readlines(), GL.GL_FRAGMENT_SHADER)

        self.shaderProgram = self.shaders.compileProgram(vertexShader, fragmentShader)
        self.shaderProgram_textureless = self.shaders.compileProgram(vertexShader_textureless, fragmentShader_textureless)
        self.shaderProgram_simple = self.shaders.compileProgram(vertexShader_simple, fragmentShader_simple)
        self.shaderProgram_material = self.shaders.compileProgram(vertexShader_material, fragmentShader_material) 
        self.shaderProgram_textureMat = self.shaders.compileProgram(vertexShader_textureMat, fragmentShader_textureMat)

        self.texUnitUniform_textureMat = GL.glGetUniformLocation(self.shaderProgram_textureMat, 'texUnit')

        self.lightpos = [0, 0, 0]

        self.fbo = GL.glGenFramebuffers(1)
        self.color_tex = GL.glGenTextures(1)
        self.color_tex_2 = GL.glGenTextures(1)
        self.color_tex_3 = GL.glGenTextures(1)
        self.color_tex_4 = GL.glGenTextures(1)
        self.color_tex_5 = GL.glGenTextures(1)

        self.depth_tex = GL.glGenTextures(1)

        GL.glBindTexture(GL.GL_TEXTURE_2D, self.color_tex)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA32F, self.width, self.height, 0,
                        GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, None)

        GL.glBindTexture(GL.GL_TEXTURE_2D, self.color_tex_2)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA32F, self.width, self.height, 0,
                        GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, None)

        GL.glBindTexture(GL.GL_TEXTURE_2D, self.color_tex_3)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA32F, self.width, self.height, 0,
                        GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, None)

        GL.glBindTexture(GL.GL_TEXTURE_2D, self.color_tex_4)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA32F, self.width, self.height, 0,
                        GL.GL_RGBA, GL.GL_FLOAT, None)

        GL.glBindTexture(GL.GL_TEXTURE_2D, self.color_tex_5)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA32F, self.width, self.height, 0,
                        GL.GL_RGBA, GL.GL_FLOAT, None)

        GL.glBindTexture(GL.GL_TEXTURE_2D, self.depth_tex)

        GL.glTexImage2D.wrappedOperation(
            GL.GL_TEXTURE_2D, 0, GL.GL_DEPTH24_STENCIL8, self.width, self.height, 0,
            GL.GL_DEPTH_STENCIL, GL.GL_UNSIGNED_INT_24_8, None)

        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.fbo)
        GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_TEXTURE_2D, self.color_tex, 0)
        GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT1, GL.GL_TEXTURE_2D, self.color_tex_2, 0)
        GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT2, GL.GL_TEXTURE_2D, self.color_tex_3, 0)
        GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT3, GL.GL_TEXTURE_2D, self.color_tex_4, 0)
        GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT4, GL.GL_TEXTURE_2D, self.color_tex_5, 0)
        GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_DEPTH_STENCIL_ATTACHMENT, GL.GL_TEXTURE_2D, self.depth_tex, 0)
        GL.glViewport(0, 0, self.width, self.height)
        GL.glDrawBuffers(5, [GL.GL_COLOR_ATTACHMENT0, GL.GL_COLOR_ATTACHMENT1,
                             GL.GL_COLOR_ATTACHMENT2, GL.GL_COLOR_ATTACHMENT3, GL.GL_COLOR_ATTACHMENT4])

        assert GL.glCheckFramebufferStatus(
            GL.GL_FRAMEBUFFER) == GL.GL_FRAMEBUFFER_COMPLETE

        self.fov = 20
        self.camera = [1, 0, 0]
        self.target = [0, 0, 0]
        self.up = [0, 0, 1]
        P = perspective(self.fov, float(self.width) /
                        float(self.height), 0.01, 100)
        V = lookat(
            self.camera,
            self.target, up=self.up)

        self.V = np.ascontiguousarray(V, np.float32)
        self.P = np.ascontiguousarray(P, np.float32)
        self.grid = self.generate_grid()
        #added mouse interaction
        self.is_rotating = False
    
    def generate_grid(self):
        VAO = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(VAO)

        vertexData = []
        for i in np.arange(-1, 1, 0.05):
            vertexData.append([i, 0, -1, 0, 0, 0, 0, 0])
            vertexData.append([i, 0, 1, 0, 0, 0, 0, 0])
            vertexData.append([1, 0, i, 0, 0, 0, 0, 0])
            vertexData.append([-1, 0, i, 0, 0, 0, 0, 0])

        vertexData = np.array(vertexData).astype(np.float32) * 3
        # Need VBO for triangle vertices and texture UV coordinates
        VBO = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, VBO)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, vertexData.nbytes, vertexData, GL.GL_STATIC_DRAW)

        # enable array and set up data
        positionAttrib = GL.glGetAttribLocation(self.shaderProgram_simple, 'position')
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(positionAttrib, 3, GL.GL_FLOAT, GL.GL_FALSE, 32, None)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
        GL.glBindVertexArray(0)

        return VAO


    def load_object(self, obj_path, texture_path, scale=1.0):

        is_materialed = True
        textures = []
        start_time = time.time()

        vertices, faces, materials, texture_paths = self.load_mesh(obj_path, scale)
        print('load mesh {:s} time:{:.3f}'.format(obj_path, time.time() - start_time))

        # compute extent
        vertices_all = vertices[0]
        for idx in range(1, len(vertices)):
            vertices_all += vertices[idx]
        extent = 2 * np.max(np.absolute(vertices_all[:, :3]), axis=0)
        self.vertices.append(vertices_all[:, :3])
        self.extents.append(extent)

        start_time = time.time()
        self.materials.append(materials)
        is_textured = []
        is_colored = []
        for texture_path in texture_paths:
            is_texture = False
            is_color = False
            if texture_path == '':
                textures.append(texture_path)
            elif texture_path == 'color':
                is_color = True
                textures.append(texture_path) 
            else:
                texture_path = os.path.join('/'.join(obj_path.split('/')[:-1]), texture_path) 
                texture = loadTexture(texture_path)
                textures.append(texture)
                is_texture = True
            is_textured.append(is_texture)
            is_colored.append(is_color)
        self.textures.append(textures)
        self.is_textured.append(is_textured)
        self.is_materialed.append(is_materialed)

        if is_materialed:# and True in is_textured: #for compatability
            for idx in range(len(vertices)):

                vertexData = vertices[idx].astype(np.float32)
                face = faces[idx]
                VAO = GL.glGenVertexArrays(1)
                GL.glBindVertexArray(VAO)

                # Need VBO for triangle vertices and texture UV coordinates
                VBO = GL.glGenBuffers(1)
                GL.glBindBuffer(GL.GL_ARRAY_BUFFER, VBO)
                GL.glBufferData(GL.GL_ARRAY_BUFFER, vertexData.nbytes, vertexData, GL.GL_STATIC_DRAW)
                if is_textured[idx]:
                    positionAttrib = GL.glGetAttribLocation(self.shaderProgram_textureMat, 'position')
                    normalAttrib = GL.glGetAttribLocation(self.shaderProgram_textureMat, 'normal')
                    coordsAttrib = GL.glGetAttribLocation(self.shaderProgram_textureMat, 'texCoords')          
                elif is_colored[idx]:
                    positionAttrib = GL.glGetAttribLocation(self.shaderProgram_textureless, 'position')
                    normalAttrib = GL.glGetAttribLocation(self.shaderProgram_textureless, 'normal')
                    colorAttrib = GL.glGetAttribLocation(self.shaderProgram_textureless, 'color')                    
                else:
                    positionAttrib = GL.glGetAttribLocation(self.shaderProgram_material, 'position')
                    normalAttrib = GL.glGetAttribLocation(self.shaderProgram_material, 'normal')   

                GL.glEnableVertexAttribArray(0)
                GL.glEnableVertexAttribArray(1)
                # the last parameter is a pointer
                if is_textured[idx]:
                    GL.glEnableVertexAttribArray(2)
                    GL.glVertexAttribPointer(positionAttrib, 3, GL.GL_FLOAT, GL.GL_FALSE, 32, None)
                    GL.glVertexAttribPointer(normalAttrib, 3, GL.GL_FLOAT, GL.GL_FALSE, 32, ctypes.c_void_p(12))
                    GL.glVertexAttribPointer(coordsAttrib, 2, GL.GL_FLOAT, GL.GL_TRUE, 32, ctypes.c_void_p(24))
                elif is_colored[idx]:
                    GL.glEnableVertexAttribArray(2)
                    GL.glVertexAttribPointer(positionAttrib, 3, GL.GL_FLOAT, GL.GL_FALSE, 36, None)
                    GL.glVertexAttribPointer(normalAttrib, 3, GL.GL_FLOAT, GL.GL_FALSE, 36, ctypes.c_void_p(12))
                    GL.glVertexAttribPointer(colorAttrib, 3, GL.GL_FLOAT, GL.GL_FALSE, 36, ctypes.c_void_p(24))                    
                else:
                    GL.glVertexAttribPointer(positionAttrib, 3, GL.GL_FLOAT, GL.GL_FALSE, 24, None)
                    GL.glVertexAttribPointer(normalAttrib, 3, GL.GL_FLOAT, GL.GL_FALSE, 24, ctypes.c_void_p(12))

                GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
                GL.glBindVertexArray(0)
                self.VAOs.append(VAO)
                self.VBOs.append(VBO)
                self.faces.append(face)
            self.objects.append(obj_path)
            self.poses_rot.append(np.eye(4))
            self.poses_trans.append(np.eye(4))  
            print('buffer time:{:.3f}'.format(time.time() - start_time))

    
    def load_offset(self):

        cur_path = os.path.abspath(os.path.dirname(__file__))
        offset_file = os.path.join(cur_path, 'robotPose', self.robot + '_models', 'center_offset.txt')
        model_file = os.path.join(cur_path, 'robotPose', self.robot + '_models', 'models.txt')
        with open(model_file, "r+") as file:
            content = file.readlines()
            model_paths = [path.strip().split('/')[-1] for path in content]
        offset = np.loadtxt(offset_file).astype(np.float32)
        offset_map = {}
        for i in range(offset.shape[0]):
            offset_map[model_paths[i]] = offset[i, :]
         #extent max - min in mesh, center = (max + min)/2
        return offset_map


    def load_mesh(self, path, scale=1.0):
        mesh_file = path.strip().split('/')[-1]  # for offset the robot mesh
        scene = load(path) #load collada
        offset = np.zeros(3)
        if self._offset_map is not None and mesh_file in self._offset_map:
            offset = self._offset_map[mesh_file]
        return self.recursive_load(scene.rootnode, [], [], [], [], offset, scale, [[], [], []])


    def recursive_load(self, node, vertices, faces, materials,
                         texture_paths, offset, scale=1, repeated=[[], [], []]):
        if node.meshes:
            transform = node.transformation 
            for idx, mesh in enumerate(node.meshes):
                if mesh.faces.shape[-1] != 3: #ignore Line Set
                    continue
                mat = mesh.material
                texture_path = False
                if hasattr(mat, 'properties'):
                    file = ('file', long(1)) if PYTHON2 else ('file', 1) 
                    if file in mat.properties:
                        texture_paths.append(mat.properties[file])
                        texture_path = True
                    else:
                        texture_paths.append('')
                mat_diffuse = np.array(mat.properties['diffuse'])[:3] 
                mat_specular = np.array(mat.properties['specular'])[:3] 
                mat_ambient = np.array(mat.properties['ambient'])[:3] #phong shader
                if 'shininess' in mat.properties:
                    mat_shininess = max(mat.properties['shininess'], 1) #avoid the 0 shininess
                else:
                    mat_shininess = 1
                mesh_vertex = homotrans(transform,mesh.vertices) - offset #subtract the offset
                if mesh.normals.shape[0] > 0:
                    mesh_normals = transform[:3,:3].dot(mesh.normals.transpose()).transpose() #normal stays the same
                else:
                    mesh_normals = np.zeros_like(mesh_vertex)
                    mesh_normals[:,-1] = 1
                if texture_path:
                    vertices.append(np.concatenate([mesh_vertex * scale, mesh_normals, mesh.texturecoords[0, :, :2]], axis=-1))
                elif mesh.colors is not None and len(mesh.colors.shape) > 2:
                    vertices.append(np.concatenate([mesh_vertex * scale, mesh_normals, mesh.colors[0, :, :3]], axis=-1)) #
                    texture_paths[-1] = 'color'
                else:
                    vertices.append(np.concatenate([mesh_vertex * scale, mesh_normals], axis=-1))
                faces.append(mesh.faces)
                materials.append(np.hstack([mat_diffuse, mat_specular, mat_ambient, mat_shininess]))
        for child in node.children:
            self.recursive_load(child, vertices, faces, materials, texture_paths, offset, scale, repeated) 
        return vertices, faces, materials, texture_paths


    def load_objects(self, obj_paths, texture_paths, colors=[[0.9, 0, 0], [0.6, 0, 0], [0.3, 0, 0]], scale=None):
        if scale is None:
            scale = [1]*len(obj_paths)
        self.colors = colors
        for i in range(len(obj_paths)):
            self.load_object(obj_paths[i], texture_paths[i], scale[i])
            if i == 0:
                self.instances.append(0)
            else:
                self.instances.append(self.instances[-1] + len(self.materials[i-1])) #offset
        print(self.extents)

    def set_camera(self, camera, target, up):
        self.camera = camera
        self.target = target
        self.up = up
        V = lookat(
            self.camera,
            self.target, up=self.up)

        self.V = np.ascontiguousarray(V, np.float32)

    def set_camera_default(self):
        self.V = np.eye(4)

    def set_fov(self, fov):
        self.fov = fov
        # this is vertical fov
        P = perspective(self.fov, float(self.width) /
                        float(self.height), 0.01, 100)
        self.P = np.ascontiguousarray(P, np.float32)


    def set_projection_matrix(self, w, h, fu, fv, u0, v0, znear, zfar):
        L = -(u0) * znear / fu;
        R = +(w-u0) * znear / fu;
        T = -(v0) * znear / fv;
        B = +(h-v0) * znear / fv;

        P = np.zeros((4, 4), dtype=np.float32);
        P[0, 0] = 2 * znear / (R-L);
        P[1, 1] = 2 * znear / (T-B);
        P[2, 0] = (R+L)/(L-R);
        P[2, 1] = (T+B)/(B-T);
        P[2, 2] = (zfar +znear) / (zfar - znear);
        P[2, 3] = 1.0;
        P[3, 2] = (2*zfar*znear)/(znear - zfar);
        self.P = P

    def set_light_color(self, color):
        self.lightcolor = color

    def render(self, cls_indexes, image_tensor, seg_tensor, normal_tensor=None, pc1_tensor=None, pc2_tensor=None):
        frame = 0
        GL.glClearColor(0, 0, 0, 1)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        GL.glEnable(GL.GL_DEPTH_TEST)
        #GL.glLightModeli(GL.GL_LIGHT_MODEL_TWO_SIDE, GL.GL_TRUE)

        if self.render_marker:
            # render some grid and directions
            GL.glUseProgram(self.shaderProgram_simple)
            GL.glBindVertexArray(self.grid)
            GL.glUniformMatrix4fv(GL.glGetUniformLocation(
                self.shaderProgram_simple, 'V'), 1, GL.GL_TRUE, self.V)
            GL.glUniformMatrix4fv(GL.glGetUniformLocation(
                self.shaderProgram_simple, 'P'), 1, GL.GL_FALSE, self.P)
            GL.glDrawElements(GL.GL_LINES, 160,
                              GL.GL_UNSIGNED_INT, np.arange(160, dtype=np.int))
            GL.glBindVertexArray(0)
            GL.glUseProgram(0)
            # end rendering markers

        size = 0
        for i in range(len(cls_indexes)):
            index = cls_indexes[i]
            is_materialed = self.is_materialed[index]
            if is_materialed:
                num = len(self.materials[index])
                for idx in range(num):
                    is_texture = self.is_textured[index][idx] #index
                    if is_texture:
                        shader = self.shaderProgram_textureMat
                    elif self.textures[index][idx] == 'color':
                        shader = self.shaderProgram_textureless
                    else:  
                        shader = self.shaderProgram_material
                    GL.glUseProgram(shader)
                    GL.glUniformMatrix4fv(GL.glGetUniformLocation(shader, 'V'), 1, GL.GL_TRUE, self.V)
                    GL.glUniformMatrix4fv(GL.glGetUniformLocation(shader, 'P'), 1, GL.GL_FALSE, self.P)
                    GL.glUniformMatrix4fv(GL.glGetUniformLocation(shader, 'pose_trans'), 1, GL.GL_FALSE, self.poses_trans[i])
                    GL.glUniformMatrix4fv(GL.glGetUniformLocation(shader, 'pose_rot'), 1, GL.GL_TRUE, self.poses_rot[i])
                    GL.glUniform3f(GL.glGetUniformLocation(shader, 'light_position'), *self.lightpos)
                    GL.glUniform3f(GL.glGetUniformLocation(shader, 'instance_color'), *self.colors[index])
                    GL.glUniform3f(GL.glGetUniformLocation(shader, 'light_color'), *self.lightcolor)

                    GL.glUniform3f(GL.glGetUniformLocation(shader, 'mat_diffuse'), *self.materials[index][idx][:3])
                    GL.glUniform3f(GL.glGetUniformLocation(shader, 'mat_specular'), *self.materials[index][idx][3:6])
                    GL.glUniform3f(GL.glGetUniformLocation(shader, 'mat_ambient'),  *self.materials[index][idx][6:9])
                    GL.glUniform1f(GL.glGetUniformLocation(shader, 'mat_shininess'), self.materials[index][idx][-1])

                    try:
                        if is_texture:
                            GL.glActiveTexture(GL.GL_TEXTURE0)
                            GL.glBindTexture(GL.GL_TEXTURE_2D, self.textures[index][idx]) #self.instances[index]
                            GL.glUniform1i(self.texUnitUniform_textureMat, 0)
                        GL.glBindVertexArray(self.VAOs[self.instances[index]+idx]) # 
                        GL.glDrawElements(GL.GL_TRIANGLES, self.faces[self.instances[index]+idx].size,
                                     GL.GL_UNSIGNED_INT, self.faces[self.instances[index]+idx])
                    finally:
                        GL.glBindVertexArray(0)
                        GL.glUseProgram(0)

        GL.glDisable(GL.GL_DEPTH_TEST)
            # mapping
        self.r.map_tensor(int(self.color_tex), int(self.width), int(self.height), image_tensor.data_ptr())
        self.r.map_tensor(int(self.color_tex_3), int(self.width), int(self.height), seg_tensor.data_ptr())
        if normal_tensor is not None:
            self.r.map_tensor(int(self.color_tex_2), int(self.width), int(self.height), normal_tensor.data_ptr())
        if pc1_tensor is not None:
            self.r.map_tensor(int(self.color_tex_4), int(self.width), int(self.height), pc1_tensor.data_ptr())
        if pc2_tensor is not None:
            self.r.map_tensor(int(self.color_tex_5), int(self.width), int(self.height), pc2_tensor.data_ptr())
             
        '''
        GL.glReadBuffer(GL.GL_COLOR_ATTACHMENT0)
        frame = GL.glReadPixels(0, 0, self.width, self.height, GL.GL_BGRA, GL.GL_FLOAT)
        #frame = np.frombuffer(frame,dtype = np.float32).reshape(self.width, self.height, 4)
        frame = frame.reshape(self.height, self.width, 4)[::-1, :]

        # GL.glReadBuffer(GL.GL_COLOR_ATTACHMENT1)
        #normal = GL.glReadPixels(0, 0, self.width, self.height, GL.GL_BGRA, GL.GL_FLOAT)
        #normal = np.frombuffer(frame, dtype=np.uint8).reshape(self.width, self.height, 4)
        #normal = normal[::-1, ]

        GL.glReadBuffer(GL.GL_COLOR_ATTACHMENT2)
        seg = GL.glReadPixels(0, 0, self.width, self.height, GL.GL_BGRA, GL.GL_FLOAT)
        #seg = np.frombuffer(frame, dtype=np.uint8).reshape(self.width, self.height, 4)
        seg = seg.reshape(self.height, self.width, 4)[::-1, :]

        #pc = GL.glReadPixels(0, 0, self.width, self.height, GL.GL_DEPTH_COMPONENT, GL.GL_FLOAT)
        # seg = np.frombuffer(frame, dtype=np.uint8).reshape(self.width, self.height, 4)

        #pc = np.stack([pc,pc, pc, np.ones(pc.shape)], axis = -1)
        #pc = pc[::-1, ]
        #pc = (1-pc) * 10

        # points in object coordinate
        GL.glReadBuffer(GL.GL_COLOR_ATTACHMENT3)
        pc2 = GL.glReadPixels(0, 0, self.width, self.height, GL.GL_RGBA, GL.GL_FLOAT)
        pc2 = pc2.reshape(self.height, self.width, 4)[::-1, :]
        pc2 = pc2[:,:,:3]

        # points in camera coordinate
        GL.glReadBuffer(GL.GL_COLOR_ATTACHMENT4)
        pc3 = GL.glReadPixels(0, 0, self.width, self.height, GL.GL_RGBA, GL.GL_FLOAT)
        pc3 = pc3.reshape(self.height, self.width, 4)[::-1, :]
        pc3 = pc3[:,:,:3]

        return [frame, seg, pc2, pc3]
        '''


    def set_light_pos(self, light):
        self.lightpos = light

    def get_num_objects(self):
        return len(self.objects)

    def set_poses(self, poses):
        self.poses_rot = [np.ascontiguousarray(
            quat2rotmat(item[3:])) for item in poses]
        self.poses_trans = [np.ascontiguousarray(
            xyz2mat(item[:3])) for item in poses]

    def set_allocentric_poses(self, poses):
        self.poses_rot = []
        self.poses_trans = []
        for pose in poses:
            x, y, z = pose[:3]
            quat_input = pose[3:]
            dx = np.arctan2(x, -z)
            dy = np.arctan2(y, -z)
            # print(dx, dy)
            quat = euler2quat(-dy, -dx, 0, axes='sxyz')
            quat = qmult(quat, quat_input)
            self.poses_rot.append(np.ascontiguousarray(quat2rotmat(quat)))
            self.poses_trans.append(np.ascontiguousarray(xyz2mat(pose[:3])))

    def release(self):
        print(self.glstring)
        self.clean()
        self.r.release()

    def clean(self):
        GL.glDeleteTextures([self.color_tex, self.color_tex_2,
                             self.color_tex_3, self.color_tex_4, self.depth_tex])
        self.color_tex = None
        self.color_tex_2 = None
        self.color_tex_3 = None
        self.color_tex_4 = None

        self.depth_tex = None
        GL.glDeleteFramebuffers(1, [self.fbo])
        self.fbo = None
        GL.glDeleteBuffers(len(self.VAOs), self.VAOs)
        self.VAOs = []
        GL.glDeleteBuffers(len(self.VBOs), self.VBOs)
        self.VBOs = []
        GL.glDeleteTextures(self.textures)
        self.textures = []
        self.objects = []  # GC should free things here
        self.faces = []  # GC should free things here
        self.poses_trans = []  # GC should free things here
        self.poses_rot = []  # GC should free things here

    def transform_vector(self, vec):
        vec = np.array(vec)
        zeros = np.zeros_like(vec)

        vec_t = self.transform_point(vec)
        zero_t = self.transform_point(zeros)

        v = vec_t - zero_t
        return v

    def transform_point(self, vec):
        vec = np.array(vec)
        if vec.shape[0] == 3:
            v = self.V.dot(np.concatenate([vec, np.array([1])]))
            return v[:3]/v[-1]
        elif vec.shape[0] == 4:
            v = self.V.dot(vec)
            return v/v[-1]
        else:
            return None

    def transform_pose(self, pose):
        pose_rot = quat2rotmat(pose[3:])
        pose_trans = xyz2mat(pose[:3])
        pose_cam = self.V.dot(pose_trans.T).dot(pose_rot).T
        return np.concatenate([mat2xyz(pose_cam), safemat2quat(pose_cam[:3, :3].T)])

    def get_num_instances(self):
        return len(self.instances)

    def get_poses(self):
        mat = [self.V.dot(self.poses_trans[i].T).dot(
            self.poses_rot[i]).T for i in range(self.get_num_instances())]
        poses = [np.concatenate(
            [mat2xyz(item), safemat2quat(item[:3, :3].T)]) for item in mat]
        return poses

    def get_egocentric_poses(self):
        return self.get_poses()

    def get_allocentric_poses(self):
        poses = self.get_poses()
        poses_allocentric = []
        for pose in poses:
            dx = np.arctan2(pose[0], -pose[2])
            dy = np.arctan2(pose[1], -pose[2])
            quat = euler2quat(-dy, -dx, 0, axes='sxyz')
            quat = qmult(qinverse(quat), pose[3:])
            poses_allocentric.append(np.concatenate([pose[:3], quat]))
            #print(quat, pose[3:], pose[:3])
        return poses_allocentric

    def get_centers(self):
        centers = []
        for i in range(len(self.poses_trans)):
            pose_trans = self.poses_trans[i]
            proj = (self.P.T.dot(self.V.dot(
                pose_trans.T).dot(np.array([0, 0, 0, 1]))))
            proj /= proj[-1]
            centers.append(proj[:2])
        centers = np.array(centers)
        centers = (centers + 1) / 2.0
        centers[:, 1] = 1 - centers[:, 1]
        centers = centers[:, ::-1]  # in y, x order
        return centers


    def vis(self, poses, cls_indexes, color_idx=None, color_list=None, cam_pos=[0, 0, 2.0], V=None,
         distance=2.0, shifted_pose=None, interact=0, window_name='test'):
        """
        a complicated visualization module 
        """
        theta = 0
        cam_x, cam_y, cam_z = cam_pos
        sample = []
        new_poses = []
        origin = np.linalg.inv(unpack_pose(poses[0]))
        if shifted_pose is not None:
            origin = np.linalg.inv(shifted_pose)
        for pose in poses:
            pose = unpack_pose(pose)
            pose = origin.dot(pose)
            new_poses.append(pack_pose(pose))
        poses = new_poses

        cam_pos = np.array([cam_x, cam_y, cam_z])  
        self.set_camera(cam_pos, cam_pos * 2 , [0, 1, 0])  
        if V is not None:
            self.V = V
            cam_pos = V[:3, 3]
        self.set_light_pos(cam_pos)      
        self.set_poses(poses)

        mouse_events = {
        'view_dir': - self.V[:3, 3],
        'view_origin': np.array([0, 0, 0.]), # anchor
        '_mouse_ix': -1,
        '_mouse_iy': -1,
        'down': False,
        'shift': False,
        'trackball': Trackball(self.width, self.height, cam_pos=cam_pos) 
        }

        image_tensor = torch.cuda.FloatTensor(self.height, self.width, 4).detach()
        seg_tensor = torch.cuda.FloatTensor(self.height, self.width, 4).detach()

        def update_dir():
            view_dir = mouse_events['view_origin'] - self.V[:3, 3]  
            self.set_camera(self.V[:3, 3], self.V[:3, 3] - view_dir, [0, 1, 0]) # would shift along the sphere
            self.V = self.V.dot(mouse_events['trackball'].property["model"].T)      
                  
        def change_dir(event, x, y, flags, param): # fix later to be a finalized version
            if event == cv2.EVENT_LBUTTONDOWN:
                mouse_events['_mouse_ix'], mouse_events['_mouse_iy'] = x, y
                mouse_events['down'] = True
            if event == cv2.EVENT_MBUTTONDOWN:
                mouse_events['_mouse_ix'], mouse_events['_mouse_iy'] = x, y
                mouse_events['shift'] = True
            if event == cv2.EVENT_MOUSEMOVE:
                if mouse_events['down']:
                    dx = (x - mouse_events['_mouse_ix']) / -10.
                    dy = (y - mouse_events['_mouse_iy']) / -10.
                    mouse_events['trackball'].on_mouse_drag(x,y,dx,dy)
                    update_dir()
                
                if mouse_events['shift']:
                    dx = (x - mouse_events['_mouse_ix']) / (-4000. / self.V[2, 3])
                    dy = (y - mouse_events['_mouse_iy']) / (-4000. / self.V[2, 3]) 
                    self.V[:3, 3] += 0.5 * np.array([dx, dy, 0])
                    mouse_events['view_origin'] += 0.5 * np.array([-dx, dy, 0]) # change
                    update_dir()
            if event == cv2.EVENT_LBUTTONUP:
                mouse_events['down'] = False
            if event == cv2.EVENT_MBUTTONUP:
                mouse_events['shift'] = False
        if interact > 0:
            cv2.namedWindow(window_name)
            cv2.setMouseCallback(window_name, change_dir)
        
        # update_dir()
        img = np.zeros([self.height, self.width, 3])
        while True:
            new_cam_pos = -self.V[:3, :3].T.dot(self.V[:3, 3])
            q = cv2.waitKey(3)
            if interact > 0:
                if q == ord('w'):
                    cam_z += 0.05
                elif q == ord('s'):
                    cam_z -= 0.05
                    interact = 2
                elif q == ord('u'):
                    interact = 1
                elif q == ord('a'):
                    theta -= 0.1
                elif q == ord('d'):
                    theta += 0.1
                elif q == ord('x'): 
                    self.V[:3, 3] += 0.02 * (self.V[:3, 3] - mouse_events['view_origin'])
                    update_dir()
                elif q == ord('c'): # move closer
                    self.V[:3, 3] -= 0.02 * (self.V[:3, 3] - mouse_events['view_origin'])
                    update_dir()
                elif q == ord('z'): # reset
                    self.set_camera(cam_pos, cam_pos * 2 , [0, 1, 0])
                    mouse_events['trackball'].reinit(cam_pos)
                    mouse_events['view_origin'] = np.zeros(3)
                elif q == ord('i'):
                    for pose in poses:
                        pose[1] += 0.02
                elif q == ord('k'):
                    for pose in poses:
                        pose[1] -= 0.02
                elif q == ord('j'):
                    for pose in poses:
                        pose[0] -= 0.02
                elif q == ord('l'):
                    for pose in poses:
                        pose[0] += 0.02
                elif q == ord('n'):
                    print('camera V', self.V)
                elif q == ord('p'):
                    cur_dir = os.path.dirname(os.path.abspath(__file__))    
                    Image.fromarray(
                    (np.clip(frame[0][:, :, [2,1,0]] * 255, 0, 255)).astype(np.uint8)).save(cur_dir + '/test.png')
                elif q == ord('q'): # wth
                    break
                elif q == ord('r'):  # rotate
                    for pose in poses:
                        pose[3:] = qmult(axangle2quat(
                            [0, 0, 1], 5/180.0 * np.pi), pose[3:])
            self.set_poses(poses)            
            self.set_light_pos(new_cam_pos) # in world coordinate
            self.render(cls_indexes, image_tensor, seg_tensor)
            image_tensor = image_tensor.flip(0)
            img = image_tensor.cpu().numpy()
            img = np.clip(img, 0, 1)
            img = img[:, :, :3] * 255
            img = img.astype(np.uint8)
            if interact > 0:
                cv2.imshow(window_name, img[:,:,::-1])
            if interact < 2:
                break
        return img

camera_extrinsics=np.array([[-0.211719, 0.97654, -0.0393032, 0.377451],[0.166697, -0.00354316, -0.986002, 0.374476],[-0.96301, -0.215307, -0.162036, 1.87315],[0,0, 0, 1]])



def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--model_path', dest='model_path',
                        help='path of the ycb models',
                        default='../data', type=str)
    parser.add_argument('--robot_name', dest='robot_name',
                        help='robot name',
                        default='', type=str)
    args = parser.parse_args()
    return args



if __name__ == '__main__':

    args = parse_args()
    model_path = args.model_path
    robot_name = args.robot_name
    width = 640
    height = 480

    renderer = YCBRenderer(width=width, height=height, render_marker=True, robot=robot_name)
    if robot_name == 'baxter':
        from robotPose.robot_pykdl import *
        print('robot name', robot_name)
        models = ['S0', 'S1', 'E0', 'E1', 'W0', 'W1', 'W2']
        #models = ['E1']
        obj_paths = [
            'robotPose/{}_models/{}.DAE'.format(robot_name,item) for item in models]
        colors = [
            [0.1*(idx+1),0,0] for idx in range(len(models))]
        texture_paths = ['' for item in models]
    elif robot_name == 'panda_arm':
        from robotPose.robot_pykdl import *
        print('robot name', robot_name)
        models = ['link1', 'link2', 'link3', 'link4', 'link5', 'link6', 'link7', 'hand', 'finger', 'finger']
        #models = ['link4']
        obj_paths = [
            'robotPose/{}_models/{}.DAE'.format(robot_name,item) for item in models]
        colors = [
            [0,0.1*(idx+1),0] for idx in range(len(models))]
        texture_paths = ['' for item in models]
    else:
        models = ["003_cracker_box", "002_master_chef_can", "011_banana"]
        colors = [[0.9, 0, 0], [0.6, 0, 0], [0.3, 0, 0]]

        obj_paths = [
            '{}/models/{}/textured_simple.obj'.format(model_path, item) for item in models]
        texture_paths = [
            '{}/models/{}/texture_map.png'.format(model_path, item) for item in models]

    print(obj_paths)
    renderer.load_objects(obj_paths, texture_paths, colors)

    # mat = pose2mat(pose)
    pose = np.array([-0.025801208, 0.08432201, 0.004528991,
                     0.9992879, -0.0021458883, 0.0304758, 0.022142926])
    pose2 = np.array([-0.56162935, 0.05060109, -0.028915625,
                      0.6582951, 0.03479896, -0.036391996, -0.75107396])
    pose3 = np.array([0.22380374, 0.019853603, 0.12159989,
                      0.9992879, -0.0021458883, 0.0304758, 0.022142926])

    theta = 0
    z = 1
    fix_pos = [np.sin(theta), z, np.cos(theta)]
    renderer.set_camera(fix_pos, [0, 0, 0], [0, 1, 0])
    fix_pos = np.zeros(3)
    renderer.set_poses([pose, pose2, pose3])
    cls_indexes = [0, 1, 2]
    if robot_name == 'baxter' or robot_name == 'panda_arm' :
        import scipy.io as sio
        robot = robot_kinematics(robot_name)
        poses = []
        if robot_name == 'baxter':
            base_link = 'right_arm_mount'
        else:
            base_link = 'panda_link0'
        pose, joint = robot.gen_rand_pose(base_link)
        cls_indexes = range(len(models)) 
        pose = robot.offset_pose_center(pose, dir='off', base_link=base_link)         #print pose_hand
        #pose = np.load('%s.npy'%robot_name) 
        for i in range(len(pose)):
            pose_i =  pose[i]
            quat = mat2quat(pose_i[:3,:3])
            trans = pose_i[:3,3]
            poses.append(np.hstack((trans,quat)))

        renderer.set_poses(poses)
        renderer.V = camera_extrinsics
        renderer.set_projection_matrix(640,480,525,525,319.5,239.5,0.0001,6) 
        fix_pos = renderer.V[:3, 3].reshape([1,3]).copy()
    renderer.set_light_pos([1, 1, 1])
    renderer.set_light_color([1.0, 1.0, 1.0])
    image_tensor = torch.cuda.FloatTensor(height, width, 4).detach()
    seg_tensor = torch.cuda.FloatTensor(height, width, 4).detach()
    
    import time
    start = time.time()
    while True:
        renderer.render(cls_indexes, image_tensor, seg_tensor)
        image_tensor = image_tensor.flip(0)
        seg_tensor = seg_tensor.flip(0)
        frame = [image_tensor.cpu().numpy(), seg_tensor.cpu().numpy()]
        centers = renderer.get_centers()
        for center in centers:
            x = int(center[1] * width)
            y = int(center[0] * height)
            frame[0][y-2:y+2, x-2:x+2, :] = 1
            frame[1][y-2:y+2, x-2:x+2, :] = 1
        if len(sys.argv) > 2 and sys.argv[2] == 'headless':
            # print(np.mean(frame[0]))
            theta += 0.001
            if theta > 1:
                break
        else:
            #import matplotlib.pyplot as plt
            #plt.imshow(np.concatenate(frame, axis=1))
            # plt.show()
            cv2.imshow('test', cv2.cvtColor(
                np.concatenate(frame, axis=1), cv2.COLOR_RGB2BGR))
            q = cv2.waitKey(16)
            if q == ord('w'):
                z += 0.05
            elif q == ord('s'):
                z -= 0.05
            elif q == ord('a'):
                theta -= 0.1
            elif q == ord('d'):
                theta += 0.1
            elif q == ord('p'):
                Image.fromarray(
                    (frame[0][:, :, :3] * 255).astype(np.uint8)).save('test.png')
            elif q == ord('q'):
                break
            elif q == ord('r'):  # rotate
                pose[3:] = qmult(axangle2quat(
                    [0, 0, 1], 5/180.0 * np.pi), pose[3:])
                pose2[3:] = qmult(axangle2quat(
                    [0, 0, 1], 5 / 180.0 * np.pi), pose2[3:])
                pose3[3:] = qmult(axangle2quat(
                    [0, 0, 1], 5 / 180.0 * np.pi), pose3[3:])
                renderer.set_poses([pose, pose2, pose3])

        cam_pos = fix_pos + np.array([np.sin(theta), z, np.cos(theta)])
        if robot_name == 'baxter' or robot_name == 'panda_arm' :
            renderer.V[:3, 3] = np.array(cam_pos)
        else:
            cam_pos = fix_pos + np.array([np.sin(theta), z, np.cos(theta)])
            renderer.set_camera(cam_pos, [0, 0, 0], [0, 1, 0])
        #renderer.set_light_pos(cam_pos)
    dt = time.time() - start
    print("{} fps".format(1000 / dt))

    renderer.release()
