import tensorflow as tf

model_fol = '/home/atsintli/Desktop/Doctorado/Models/saved_models/Essentia_MFCCs_Loudness/'
output_graph = 'TEST_MODEL_FROZEN.pb'


with tf.name_scope('model'):
    DEFINE_YOUR_ARCHITECTURE()

sess = tf.Session()
sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()
saver.restore(sess, '/home/atsintli/Desktop/Doctorado/Models/training_1/')

gd = sess.graph.as_graph_def()
for node in gd.node:
    if node.op == 'RefSwitch':
        node.op = 'Switch'
        for index in range(len(node.input)):
            if 'moving_' in node.input[index]:
                node.input[index] = node.input[index] + '/read'
    elif node.op == 'AssignSub':
        node.op = 'Sub'
        if 'use_locking' in node.attr: del node.attr['use_locking']
    elif node.op == 'AssignAdd':
        node.op = 'Add'
        if 'use_locking' in node.attr: del node.attr['use_locking']
    elif node.op == 'Assign':
        node.op = 'Identity'
        if 'use_locking' in node.attr: del node.attr['use_locking']
        if 'validate_shape' in node.attr: del node.attr['validate_shape']
        if len(node.input) == 2:
            node.input[0] = node.input[1]
            del node.input[1]

node_names =[n.name for n in gd.node]

output_graph_def = tf.graph_util.convert_variables_to_constants(
    sess, gd, node_names)

# Write to Protobuf format
tf.io.write_graph(output_graph_def, model_fol, output_graph, as_text=False)
sess.close()
