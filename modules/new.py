import tensorflow as tf
# from google.protobuf import text_format
#
# with open('./tmp/mnist_convnet_model/graph.pbtxt') as f:
#     txt = f.read()
# gdef = text_format.Parse(txt, tf.GraphDef())
#
# tf.train.write_graph(gdef, './tmp', 'graph.pb', as_text=False)
# gf = tf.GraphDef()
# a = gf.ParseFromString(open('/media/basanta/main/academics/Nepscan/tmp/mnist_convnet_model/frozen_model.pb','rb').read())
from tensorflow.python.framework import tensor_util

with tf.gfile.GFile('/media/basanta/main/academics/Nepscan/tmp/mnist_convnet_model/frozen_model.pb', "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

# import graph_def
with tf.Graph().as_default() as graph:
    tf.import_graph_def(graph_def, name='')
#     graph_nodes=[n for n in graph_def.node]
#
# wts = [n for n in graph_nodes if n.op=='Const']
# for n in wts:
#     print "Name of the node - %s" % n.name
#     print "Value - "
#     print tensor_util.MakeNdarray(n.attr['value'].tensor)

# print operations
for op in graph.get_operations():
    print("op.name : ",op.name)
    print ("op.values : ",op.values)

print ("----------------------------")
tensor_names = [t.name for op in graph.get_operations() for t in op.values()]
# print tensor_names
#
# a = [n.name + '=>' + n.op for n in graph_def.node if n.op in ('Softmax','Mul')]
# print a