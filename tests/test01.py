from gbo.core import Node, Segment, VascularTree, TissueGrid
tree = VascularTree()
r = 0.5
n0 = tree.add_node([0,0,0], r)
n1 = tree.add_node([1,0,0], r, parent=n0)
seg = next(iter(tree.edges()))
print(round(seg.length, 1))