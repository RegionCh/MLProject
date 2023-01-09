import torch

#initialize parameters(w,b)
def initialize_parameters(layer_dims):
    """
    :param layer_dims: list,每一层单元的个数（维度）
	:return:dictionary,存储参数w1,w2,...,wL,b1,...,bL
	"""
    L = len(layer_dims)#the number of layers in the network
    parameters = {}
    for l in range(1,L):
        parameters["W" + str(l)] = torch.randn(layer_dims[l],layer_dims[l-1])*0.1
        parameters["b" + str(l)] = torch.zeros((layer_dims[l],1))
    return parameters

def linear_forward(x, w, b):
	"""
	:param x:
	:param w:
	:param b:
	:return:
	"""
	w=w.to(torch.float32)
	x=x.to(torch.float32)
	z = torch.mm(w, x) + b  # 计算z = wx + b
	return z

#implement the activation function(ReLU and sigmoid)

def relu_forward(Z):
    # """
    # :param Z: Output of the activation layer
    # :return:
    # A: output of activation
    # """
	row = Z.size(0)
	col = Z.size(1)
	A = torch.maximum(Z,torch.zeros(row,col))
	return A

def sigmoid(Z):
    """
	:param Z: Output of the linear layer
	:return:
	"""
    A = 1 / (1 + torch.exp(-Z))
    return A

def forward_propagation(X, parameters):
	"""
	X -- input dataset, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2",...,"WL", "bL"
                    W -- weight matrix of shape (size of current layer, size of previous layer)
                    b -- bias vector of shape (size of current layer,1)
    :return:
	AL: the output of the last Layer(y_predict)
	caches: list, every element is a tuple:(W,b,z,A_pre)
	"""
	L = len(parameters) // 2  # number of layer
	A = X
	caches = []
	# calculate from 1 to L-1 layer
	for l in range(1,L):
		W = parameters["W" + str(l)]
		b = parameters["b" + str(l)]
		#linear forward -> relu forward ->linear forward....
		z = linear_forward(A, W, b)
		caches.append((A, W, b, z))  # 以激活函数为分割，到z认为是这一层的，激活函数的输出值A认为是下一层的输入，划归到下一层。注意cache的位置，要放在relu前面。
		A = relu_forward(z) #relu activation function
	# calculate Lth layer
	WL = parameters["W" + str(L)]
	bL = parameters["b" + str(L)]
	zL = linear_forward(A, WL, bL)
	caches.append((A, WL, bL, zL))
	AL = sigmoid(zL)
	return AL, caches

#calculate cost function
def compute_cost(AL,Y):
	"""
	:param AL: 最后一层的激活值,即预测值,shape:(1,number of examples)
	:param Y:真实值,shape:(1, number of examples)
	:return:
	"""
	m = Y.shape[1]
	cost = 1. / m * torch.nansum(torch.mul(-torch.log(AL), Y) +
	                          torch.mul(-torch.log(1 - AL), 1 - Y))
	#从数组的形状中删除单维条目，即把shape中为1的维度去掉，比如把[[[2]]]变成2
	cost = torch.squeeze(cost)
	return cost

#derivation of relu
def relu_backward(dA, Z):
	"""
	:param Z: the input of activation function
	:param dA:
	:return:
	"""
	Z = torch.where(Z < 0, 0, Z)
	Z=Z.to(torch.float32)
	dout = torch.mul(dA,Z) #J对z的求导
	return dout

#derivation of linear
def linear_backward(dZ, cache):
	"""
	:param dZ: Upstream derivative, the shape (n^[l+1],m)
	:param A: input of this layer
	:return:
	"""
	A, W, b, z = cache
	dZ=dZ.to(torch.float32)
	A=A.to(torch.float32)
	dW = torch.mm(dZ, A.T)
	db = torch.sum(dZ, axis=1, keepdims=True)
	da = torch.mm(W.T, dZ)
	return da, dW, db


def backward_propagation(AL, Y, caches):
	"""
	Implement the backward propagation presented in figure 2.
	Arguments:
	X -- input dataset, of shape (input size, number of examples)
	Y -- true "label" vector (containing 0 if cat, 1 if non-cat)
	caches -- caches output from forward_propagation(),(W,b,z,pre_A)

	Returns:
	gradients -- A dictionary with the gradients with respect to dW,db
	"""
	m = Y.shape[1]
	L = len(caches) - 1
	#calculate the Lth layer gradients
	dz = 1. / m * (AL - Y)
	da, dWL, dbL = linear_backward(dz, caches[L])
	gradients = {"dW" + str(L + 1): dWL, "db" + str(L + 1): dbL}

	#calculate from L-1 to 1 layer gradients
	for l in reversed(range(0,L)): # L-1,L-3,....,0
		A, W, b, z = caches[l]
		#ReLu backward -> linear backward
		#relu backward
		dout = relu_backward(da, z)
		#linear backward
		da, dW, db = linear_backward(dout, caches[l])
		# print("========dW" + str(l+1) + "================")
		# print(dW.shape)
		gradients["dW" + str(l+1)] = dW
		gradients["db" + str(l+1)] = db
	return gradients

def update_parameters(parameters, grads, learning_rate):
	"""
	:param parameters: dictionary,  W,b
	:param grads: dW,db
	:param learning_rate: alpha
	:return:
	"""
	L = len(parameters) 
	for l in range(L):
		parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l+1)]
		parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l+1)]
	return parameters

def L_layer_model(X, Y, layer_dims, learning_rate, num_iterations):
	"""
	:param X:
	:param Y:
	:param layer_dims:list containing the input size and each layer size
	:param learning_rate:
	:param num_iterations:
	:return:
	parameters：final parameters:(W,b)
	"""
	costs = []
	# initialize parameters
	parameters = initialize_parameters(layer_dims)
	for i in range(0, num_iterations):
		#foward propagation
		AL,caches = forward_propagation(X, parameters)
		# calculate the cost
		cost = compute_cost(AL, Y)
		if i % 200 == 0:
			print("Cost after iteration {}: {}".format(i, cost))
			costs.append(cost)
		#backward propagation
		grads = backward_propagation(AL, Y, caches)
		#update parameters
		parameters = update_parameters(parameters, grads, learning_rate)
	print('length of cost')
	print(len(costs))
	return parameters

#predict function
def predict(X_test,y_test,parameters):
	"""
	:param X:
	:param y:
	:param parameters:
	:return:
	"""
	correct=0
	m = y_test.shape[1]
	prob, caches = forward_propagation(X_test,parameters)
	for i in range(prob.shape[1]):
		# Convert probabilities A[0,i] to actual predictions p[0,i]
		if abs(prob[0, i]-y_test[0,i]) <= 1:
			correct+=1
	accuracy = correct/m
	return accuracy

#DNN model
def DNN(X_train, y_train, X_test, y_test, layer_dims, learning_rate= 0.001, num_iterations=30000):
	parameters = L_layer_model(X_train, y_train, layer_dims, learning_rate, num_iterations)
	accuracy = predict(X_test,y_test,parameters)
	return accuracy
