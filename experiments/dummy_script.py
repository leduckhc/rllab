

import theano
import theano.tensor as T
import numpy


qvals = T.matrix('qvals')
actions = T.vector('actions',dtype='int32')
n_steps = T.iscalar('n_steps')

def set_qvals(a_qval, a_acti):
    return a_qval[a_acti]


scan_result, scan_updates = theano.scan(fn=set_qvals, sequences=[qvals, actions], n_steps=n_steps)
f_get_qvals = theano.function(inputs=[qvals, actions, n_steps], outputs=scan_result)

test_qvals = numpy.array([[1,2.3],[1.5,3.9],[5.7,1.3]])
test_actions = numpy.array([1, 1, 0], dtype=numpy.int32)
test_n_steps = 3

print(f_get_qvals(test_qvals, test_actions, test_n_steps))



result = qvals[T.arange(qvals.shape[0]), actions]
fuck_get_qvals = theano.function(inputs=[qvals, actions], outputs=result)
print(fuck_get_qvals(test_qvals, test_actions))

exit()