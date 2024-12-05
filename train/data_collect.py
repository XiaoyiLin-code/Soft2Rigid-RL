# -*- coding: utf-8 -*-
import Sofa.Core
import Sofa
from stlib3.physics.deformable import ElasticMaterialObject
from stlib3.physics.constraints import FixedBox
from softrobots.actuators import PullingCable
from stlib3.physics.collision import CollisionMesh
from splib3.loaders import loadPointListFromFile
import json
import Sofa.constants.Key as Key
# 文件处理类，用于保存数据
class FileHandler(Sofa.Core.Controller):
    def __init__(self, *args, **kwargs):
        Sofa.Core.Controller.__init__(self, *args, **kwargs)
        self.filename = args[0]

    def write_file(self, data):
        with open(self.filename, 'w') as file_object:
            json.dump(data, file_object, indent=4)

# 自动手指控制器
class AutoFingerController(Sofa.Core.Controller):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cable = args[0]
        self.node = args[1]
        self.name = "AutoFingerController"
        self.displacement = 0.0
        self.data = {"displacement_position_map": []}
        self.file_handler = FileHandler("displacement_position.json")

    def onBeginAnimationStep(self, dt):
        if not hasattr(self.cable, 'CableConstraint'):
            print("Error: CableConstraint is not found!")
            return

        self.displacement += 0.2
        self.cable.CableConstraint.value = [self.displacement]

        if hasattr(self.node, 'Markers') and hasattr(self.node.Markers, 'MechanicalObject'):
            positions = self.node.Markers.MechanicalObject.findData('position').value
            positions_list = [[p[0], p[1], p[2]] for p in positions]
            self.data["displacement_position_map"].append({
                "displacement": self.displacement,
                "positions": positions_list
            })

    def onEndAnimationStep(self):
        if self.file_handler:
            self.file_handler.write_file(self.data)

class FingerController(Sofa.Core.Controller):

    def __init__(self, *args, **kwargs):
        Sofa.Core.Controller.__init__(self, args, kwargs)
        self.cable = args[0]
        self.node = args[1]
        self.name = "FingerController"
        self.displacement = 0.0
        self.data = {"displacement_position_map": []}
        self.file_handler = FileHandler("./displacement_position1.json")


    def onKeypressedEvent(self, e):
        self.displacement = self.cable.CableConstraint.value[0]
        if e["key"] == Key.plus:
            self.displacement += 0.05
            self.cable.CableConstraint.value = [self.displacement]
            for i in range(0,6,2):
                position = self.node.Markers.MechanicalObject.findData('position').value[i]
                print(f"marker{i + 1}:{self.displacement},{position}", end=" ")

            if hasattr(self.node, 'Markers') and hasattr(self.node.Markers, 'MechanicalObject'):
                positions = self.node.Markers.MechanicalObject.findData('position').value
                positions_list = [[p[0], p[1], p[2]] for p in positions]
                self.data["displacement_position_map"].append({
                    "displacement": self.displacement,
                    "positions": positions_list
                })
        elif e["key"] == Key.uparrow:
            if self.file_handler:
              self.file_handler.write_file(self.data)

# 手指模型定义
def Finger(parentNode=None, name="Finger",
           rotation=[0.0, 0.0, 0.0], translation=[0.0, 0.0, 0.0],
           fixingBox=[-10, -20, -20, 10, 0, 3], pullPointLocation=[0.0, -20, 0.0]):
    finger = parentNode.addChild(name)

    eobject = ElasticMaterialObject(finger,
                                    volumeMeshFileName="data/mesh/index.vtk",
                                    poissonRatio=0.3,
                                    youngModulus=40000,
                                    totalMass=0.5,
                                    surfaceColor=[0.0, 0.8, 0.7, 1.0],
                                    surfaceMeshFileName="data/mesh/index.stl",
                                    rotation=rotation,
                                    translation=translation)

    finger.addChild(eobject)
    FixedBox(eobject, atPositions=fixingBox, doVisualization=True)

    markers = eobject.addChild('Markers')
    markers.addObject('MechanicalObject', template="Vec3", position=[[4.85, 6, 2], [4.85, 16, 2], [4.85, 26, 2],
                                                                     [4.85, 36, 2], [4.85, 46, 2], [4.85, 56, 2]],
                      showIndices=True)
    markers.addObject('BarycentricMapping')

    cable = PullingCable(eobject,
                         "PullingCable",
                         pullPointLocation=pullPointLocation,
                         rotation=rotation,
                         translation=translation,
                         cableGeometry=loadPointListFromFile("data/mesh/cable_nzy.json"))

    eobject.addObject(FingerController(cable, eobject))

    CollisionMesh(eobject, name="CollisionMesh",
                  surfaceMeshFileName="/home/lightcone/workspace/SOFA/hand/SoftHand/data/finger1.stl",
                  rotation=rotation, translation=translation,
                  collisionGroup=[1, 2])

    return finger

# 场景创建函数
def createScene(rootNode):
    # 加载必要插件
    rootNode.addObject('RequiredPlugin', name='Sofa.Component.AnimationLoop')
    rootNode.addObject('RequiredPlugin', name='Sofa.Component.Collision.Detection.Algorithm')
    rootNode.addObject('RequiredPlugin', name='Sofa.Component.Collision.Detection.Intersection')
    rootNode.addObject('RequiredPlugin', name='Sofa.Component.Collision.Geometry')
    rootNode.addObject('RequiredPlugin', name='Sofa.Component.Collision.Response.Contact')
    rootNode.addObject('RequiredPlugin', name='Sofa.Component.Constraint.Lagrangian.Correction')
    rootNode.addObject('RequiredPlugin', name='Sofa.Component.Constraint.Lagrangian.Solver')
    rootNode.addObject('RequiredPlugin', name='Sofa.Component.Engine.Select')
    rootNode.addObject('RequiredPlugin', name='Sofa.Component.LinearSolver.Direct')
    rootNode.addObject('RequiredPlugin', name='Sofa.Component.Mapping.Linear')
    rootNode.addObject('RequiredPlugin', name='Sofa.Component.Mass')
    rootNode.addObject('RequiredPlugin', name='Sofa.Component.ODESolver.Backward')
    rootNode.addObject('RequiredPlugin', name='Sofa.Component.SolidMechanics.FEM.Elastic')
    rootNode.addObject('RequiredPlugin', name='Sofa.Component.SolidMechanics.Spring')
    rootNode.addObject('RequiredPlugin', name='Sofa.Component.StateContainer')
    rootNode.addObject('RequiredPlugin', name='Sofa.Component.Topology.Container.Constant')
    rootNode.addObject('RequiredPlugin', name='Sofa.Component.Topology.Container.Dynamic')
    rootNode.addObject('RequiredPlugin', name='Sofa.Component.Visual')

    rootNode.addObject('FreeMotionAnimationLoop')
    rootNode.dt = 0.01

    from stlib3.scene import MainHeader, ContactHeader
    MainHeader(rootNode, gravity=[0.0, -981.0, 0.0], plugins=["SoftRobots"])
    ContactHeader(rootNode, alarmDistance=4, contactDistance=3, frictionCoef=0.08)
    rootNode.VisualStyle.displayFlags = "showBehavior showCollisionModels"

    Finger(rootNode, name="Finger",
           rotation=[0.0, 0.0, 0.0],
           translation=[0.0, 0.0, 0.0],
           fixingBox=[-10, -20, -20, 10, 0, 3],
           pullPointLocation=[0.0, -20, 0.0])

    return rootNode
