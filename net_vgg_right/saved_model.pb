��
��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
�
Conv3D

input"T
filter"T
output"T"
Ttype:
2"
strides	list(int)(0""
paddingstring:
SAMEVALID"0
data_formatstringNDHWC:
NDHWCNCDHW"!
	dilations	list(int)	

.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
�
	MaxPool3D

input"T
output"T"
ksize	list(int)(0"
strides	list(int)(0""
paddingstring:
SAMEVALID"0
data_formatstringNDHWC:
NDHWCNCDHW"
Ttype:
2
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(�

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.9.02v2.9.0-rc2-42-g8a20d54a3c18��
z
Adam/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/dense/bias/v
s
%Adam/dense/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense/bias/v*
_output_shapes
:*
dtype0
�
Adam/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�	*$
shared_nameAdam/dense/kernel/v
|
'Adam/dense/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/v*
_output_shapes
:	�	*
dtype0
z
Adam/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/dense/bias/m
s
%Adam/dense/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense/bias/m*
_output_shapes
:*
dtype0
�
Adam/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�	*$
shared_nameAdam/dense/kernel/m
|
'Adam/dense/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/m*
_output_shapes
:	�	*
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:*
dtype0
u
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�	*
shared_namedense/kernel
n
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes
:	�	*
dtype0
s
conv3d_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nameconv3d_7/bias
l
!conv3d_7/bias/Read/ReadVariableOpReadVariableOpconv3d_7/bias*
_output_shapes	
:�*
dtype0
�
conv3d_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*!
shape:��* 
shared_nameconv3d_7/kernel
�
#conv3d_7/kernel/Read/ReadVariableOpReadVariableOpconv3d_7/kernel*,
_output_shapes
:��*
dtype0
s
conv3d_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nameconv3d_6/bias
l
!conv3d_6/bias/Read/ReadVariableOpReadVariableOpconv3d_6/bias*
_output_shapes	
:�*
dtype0
�
conv3d_6/kernelVarHandleOp*
_output_shapes
: *
dtype0* 
shape:@�* 
shared_nameconv3d_6/kernel
�
#conv3d_6/kernel/Read/ReadVariableOpReadVariableOpconv3d_6/kernel*+
_output_shapes
:@�*
dtype0
r
conv3d_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv3d_5/bias
k
!conv3d_5/bias/Read/ReadVariableOpReadVariableOpconv3d_5/bias*
_output_shapes
:@*
dtype0
�
conv3d_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@* 
shared_nameconv3d_5/kernel

#conv3d_5/kernel/Read/ReadVariableOpReadVariableOpconv3d_5/kernel**
_output_shapes
:@@*
dtype0
r
conv3d_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv3d_4/bias
k
!conv3d_4/bias/Read/ReadVariableOpReadVariableOpconv3d_4/bias*
_output_shapes
:@*
dtype0
�
conv3d_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @* 
shared_nameconv3d_4/kernel

#conv3d_4/kernel/Read/ReadVariableOpReadVariableOpconv3d_4/kernel**
_output_shapes
: @*
dtype0
r
conv3d_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv3d_3/bias
k
!conv3d_3/bias/Read/ReadVariableOpReadVariableOpconv3d_3/bias*
_output_shapes
: *
dtype0
�
conv3d_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  * 
shared_nameconv3d_3/kernel

#conv3d_3/kernel/Read/ReadVariableOpReadVariableOpconv3d_3/kernel**
_output_shapes
:  *
dtype0
r
conv3d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv3d_2/bias
k
!conv3d_2/bias/Read/ReadVariableOpReadVariableOpconv3d_2/bias*
_output_shapes
: *
dtype0
�
conv3d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_nameconv3d_2/kernel

#conv3d_2/kernel/Read/ReadVariableOpReadVariableOpconv3d_2/kernel**
_output_shapes
: *
dtype0
r
conv3d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv3d_1/bias
k
!conv3d_1/bias/Read/ReadVariableOpReadVariableOpconv3d_1/bias*
_output_shapes
:*
dtype0
�
conv3d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv3d_1/kernel

#conv3d_1/kernel/Read/ReadVariableOpReadVariableOpconv3d_1/kernel**
_output_shapes
:*
dtype0
n
conv3d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv3d/bias
g
conv3d/bias/Read/ReadVariableOpReadVariableOpconv3d/bias*
_output_shapes
:*
dtype0
�
conv3d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv3d/kernel
{
!conv3d/kernel/Read/ReadVariableOpReadVariableOpconv3d/kernel**
_output_shapes
:*
dtype0

NoOpNoOp
�l
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�k
value�kB�k B�k
�
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer-6
layer_with_weights-4
layer-7
	layer_with_weights-5
	layer-8

layer-9
layer_with_weights-6
layer-10
layer_with_weights-7
layer-11
layer-12
layer-13
layer_with_weights-8
layer-14
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer
loss

signatures*
'
#_self_saveable_object_factories* 
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
* &call_and_return_all_conditional_losses

!kernel
"bias
##_self_saveable_object_factories
 $_jit_compiled_convolution_op*
�
%	variables
&trainable_variables
'regularization_losses
(	keras_api
)__call__
**&call_and_return_all_conditional_losses

+kernel
,bias
#-_self_saveable_object_factories
 ._jit_compiled_convolution_op*
�
/	variables
0trainable_variables
1regularization_losses
2	keras_api
3__call__
*4&call_and_return_all_conditional_losses
#5_self_saveable_object_factories* 
�
6	variables
7trainable_variables
8regularization_losses
9	keras_api
:__call__
*;&call_and_return_all_conditional_losses

<kernel
=bias
#>_self_saveable_object_factories
 ?_jit_compiled_convolution_op*
�
@	variables
Atrainable_variables
Bregularization_losses
C	keras_api
D__call__
*E&call_and_return_all_conditional_losses

Fkernel
Gbias
#H_self_saveable_object_factories
 I_jit_compiled_convolution_op*
�
J	variables
Ktrainable_variables
Lregularization_losses
M	keras_api
N__call__
*O&call_and_return_all_conditional_losses
#P_self_saveable_object_factories* 
�
Q	variables
Rtrainable_variables
Sregularization_losses
T	keras_api
U__call__
*V&call_and_return_all_conditional_losses

Wkernel
Xbias
#Y_self_saveable_object_factories
 Z_jit_compiled_convolution_op*
�
[	variables
\trainable_variables
]regularization_losses
^	keras_api
___call__
*`&call_and_return_all_conditional_losses

akernel
bbias
#c_self_saveable_object_factories
 d_jit_compiled_convolution_op*
�
e	variables
ftrainable_variables
gregularization_losses
h	keras_api
i__call__
*j&call_and_return_all_conditional_losses
#k_self_saveable_object_factories* 
�
l	variables
mtrainable_variables
nregularization_losses
o	keras_api
p__call__
*q&call_and_return_all_conditional_losses

rkernel
sbias
#t_self_saveable_object_factories
 u_jit_compiled_convolution_op*
�
v	variables
wtrainable_variables
xregularization_losses
y	keras_api
z__call__
*{&call_and_return_all_conditional_losses

|kernel
}bias
#~_self_saveable_object_factories
 _jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
$�_self_saveable_object_factories* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
$�_self_saveable_object_factories* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias*
�
!0
"1
+2
,3
<4
=5
F6
G7
W8
X9
a10
b11
r12
s13
|14
}15
�16
�17*

�0
�1*
B
�0
�1
�2
�3
�4
�5
�6
�7* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
:
�trace_0
�trace_1
�trace_2
�trace_3* 
:
�trace_0
�trace_1
�trace_2
�trace_3* 
* 
u
	�iter
�beta_1
�beta_2

�decay
�learning_rate	�m�	�m�	�v�	�v�*

�trace_0
�trace_1* 

�serving_default* 
* 

!0
"1*
* 


�0* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
__call__
* &call_and_return_all_conditional_losses
& "call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
]W
VARIABLE_VALUEconv3d/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEconv3d/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 

+0
,1*
* 


�0* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
%	variables
&trainable_variables
'regularization_losses
)__call__
**&call_and_return_all_conditional_losses
&*"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
_Y
VARIABLE_VALUEconv3d_1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEconv3d_1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
/	variables
0trainable_variables
1regularization_losses
3__call__
*4&call_and_return_all_conditional_losses
&4"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
* 

<0
=1*
* 


�0* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
6	variables
7trainable_variables
8regularization_losses
:__call__
*;&call_and_return_all_conditional_losses
&;"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
_Y
VARIABLE_VALUEconv3d_2/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEconv3d_2/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 

F0
G1*
* 


�0* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
@	variables
Atrainable_variables
Bregularization_losses
D__call__
*E&call_and_return_all_conditional_losses
&E"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
_Y
VARIABLE_VALUEconv3d_3/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEconv3d_3/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
J	variables
Ktrainable_variables
Lregularization_losses
N__call__
*O&call_and_return_all_conditional_losses
&O"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
* 

W0
X1*
* 


�0* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
Q	variables
Rtrainable_variables
Sregularization_losses
U__call__
*V&call_and_return_all_conditional_losses
&V"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
_Y
VARIABLE_VALUEconv3d_4/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEconv3d_4/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 

a0
b1*
* 


�0* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
[	variables
\trainable_variables
]regularization_losses
___call__
*`&call_and_return_all_conditional_losses
&`"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
_Y
VARIABLE_VALUEconv3d_5/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEconv3d_5/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
e	variables
ftrainable_variables
gregularization_losses
i__call__
*j&call_and_return_all_conditional_losses
&j"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
* 

r0
s1*
* 


�0* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
l	variables
mtrainable_variables
nregularization_losses
p__call__
*q&call_and_return_all_conditional_losses
&q"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
_Y
VARIABLE_VALUEconv3d_6/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEconv3d_6/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 

|0
}1*
* 


�0* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
v	variables
wtrainable_variables
xregularization_losses
z__call__
*{&call_and_return_all_conditional_losses
&{"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
_Y
VARIABLE_VALUEconv3d_7/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEconv3d_7/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
\V
VARIABLE_VALUEdense/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUE
dense/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE*

�trace_0* 

�trace_0* 

�trace_0* 

�trace_0* 

�trace_0* 

�trace_0* 

�trace_0* 

�trace_0* 
z
!0
"1
+2
,3
<4
=5
F6
G7
W8
X9
a10
b11
r12
s13
|14
}15*
r
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14*

�0
�1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 

!0
"1*
* 
* 


�0* 
* 
* 
* 

+0
,1*
* 
* 


�0* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

<0
=1*
* 
* 


�0* 
* 
* 
* 

F0
G1*
* 
* 


�0* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

W0
X1*
* 
* 


�0* 
* 
* 
* 

a0
b1*
* 
* 


�0* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

r0
s1*
* 
* 


�0* 
* 
* 
* 

|0
}1*
* 
* 


�0* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<
�	variables
�	keras_api

�total

�count*
M
�	variables
�	keras_api

�total

�count
�
_fn_kwargs*

�0
�1*

�	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

�0
�1*

�	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
y
VARIABLE_VALUEAdam/dense/kernel/mRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/dense/bias/mPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense/kernel/vRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/dense/bias/vPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�
serving_default_input_1Placeholder*3
_output_shapes!
:���������22*
dtype0*(
shape:���������22
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1conv3d/kernelconv3d/biasconv3d_1/kernelconv3d_1/biasconv3d_2/kernelconv3d_2/biasconv3d_3/kernelconv3d_3/biasconv3d_4/kernelconv3d_4/biasconv3d_5/kernelconv3d_5/biasconv3d_6/kernelconv3d_6/biasconv3d_7/kernelconv3d_7/biasdense/kernel
dense/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*4
_read_only_resource_inputs
	
*4
config_proto$"

CPU

GPU(2*0J 8� *,
f'R%
#__inference_signature_wrapper_11402
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename!conv3d/kernel/Read/ReadVariableOpconv3d/bias/Read/ReadVariableOp#conv3d_1/kernel/Read/ReadVariableOp!conv3d_1/bias/Read/ReadVariableOp#conv3d_2/kernel/Read/ReadVariableOp!conv3d_2/bias/Read/ReadVariableOp#conv3d_3/kernel/Read/ReadVariableOp!conv3d_3/bias/Read/ReadVariableOp#conv3d_4/kernel/Read/ReadVariableOp!conv3d_4/bias/Read/ReadVariableOp#conv3d_5/kernel/Read/ReadVariableOp!conv3d_5/bias/Read/ReadVariableOp#conv3d_6/kernel/Read/ReadVariableOp!conv3d_6/bias/Read/ReadVariableOp#conv3d_7/kernel/Read/ReadVariableOp!conv3d_7/bias/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp'Adam/dense/kernel/m/Read/ReadVariableOp%Adam/dense/bias/m/Read/ReadVariableOp'Adam/dense/kernel/v/Read/ReadVariableOp%Adam/dense/bias/v/Read/ReadVariableOpConst*,
Tin%
#2!	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8� *'
f"R 
__inference__traced_save_12254
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv3d/kernelconv3d/biasconv3d_1/kernelconv3d_1/biasconv3d_2/kernelconv3d_2/biasconv3d_3/kernelconv3d_3/biasconv3d_4/kernelconv3d_4/biasconv3d_5/kernelconv3d_5/biasconv3d_6/kernelconv3d_6/biasconv3d_7/kernelconv3d_7/biasdense/kernel
dense/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotal_1count_1totalcountAdam/dense/kernel/mAdam/dense/bias/mAdam/dense/kernel/vAdam/dense/bias/v*+
Tin$
"2 *
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8� **
f%R#
!__inference__traced_restore_12357��
�

�
__inference_loss_fn_2_12083O
1kernel_regularizer_square_readvariableop_resource: 
identity��(kernel/Regularizer/Square/ReadVariableOp�
(kernel/Regularizer/Square/ReadVariableOpReadVariableOp1kernel_regularizer_square_readvariableop_resource**
_output_shapes
: *
dtype0�
kernel/Regularizer/SquareSquare0kernel/Regularizer/Square/ReadVariableOp:value:0*
T0**
_output_shapes
: u
kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
kernel/Regularizer/SumSumkernel/Regularizer/Square:y:0!kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: ]
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;�
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: X
IdentityIdentitykernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: q
NoOpNoOp)^kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2T
(kernel/Regularizer/Square/ReadVariableOp(kernel/Regularizer/Square/ReadVariableOp
�
�
(__inference_conv3d_4_layer_call_fn_11905

inputs%
unknown: @
	unknown_0:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������@*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *L
fGRE
C__inference_conv3d_4_layer_call_and_return_conditional_losses_10573{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:��������� : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:��������� 
 
_user_specified_nameinputs
�

�
__inference_loss_fn_0_12061O
1kernel_regularizer_square_readvariableop_resource:
identity��(kernel/Regularizer/Square/ReadVariableOp�
(kernel/Regularizer/Square/ReadVariableOpReadVariableOp1kernel_regularizer_square_readvariableop_resource**
_output_shapes
:*
dtype0�
kernel/Regularizer/SquareSquare0kernel/Regularizer/Square/ReadVariableOp:value:0*
T0**
_output_shapes
:u
kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
kernel/Regularizer/SumSumkernel/Regularizer/Square:y:0!kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: ]
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;�
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: X
IdentityIdentitykernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: q
NoOpNoOp)^kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2T
(kernel/Regularizer/Square/ReadVariableOp(kernel/Regularizer/Square/ReadVariableOp
֗
�
@__inference_model_layer_call_and_return_conditional_losses_11772

inputsC
%conv3d_conv3d_readvariableop_resource:4
&conv3d_biasadd_readvariableop_resource:E
'conv3d_1_conv3d_readvariableop_resource:6
(conv3d_1_biasadd_readvariableop_resource:E
'conv3d_2_conv3d_readvariableop_resource: 6
(conv3d_2_biasadd_readvariableop_resource: E
'conv3d_3_conv3d_readvariableop_resource:  6
(conv3d_3_biasadd_readvariableop_resource: E
'conv3d_4_conv3d_readvariableop_resource: @6
(conv3d_4_biasadd_readvariableop_resource:@E
'conv3d_5_conv3d_readvariableop_resource:@@6
(conv3d_5_biasadd_readvariableop_resource:@F
'conv3d_6_conv3d_readvariableop_resource:@�7
(conv3d_6_biasadd_readvariableop_resource:	�G
'conv3d_7_conv3d_readvariableop_resource:��7
(conv3d_7_biasadd_readvariableop_resource:	�7
$dense_matmul_readvariableop_resource:	�	3
%dense_biasadd_readvariableop_resource:
identity��conv3d/BiasAdd/ReadVariableOp�conv3d/Conv3D/ReadVariableOp�conv3d_1/BiasAdd/ReadVariableOp�conv3d_1/Conv3D/ReadVariableOp�conv3d_2/BiasAdd/ReadVariableOp�conv3d_2/Conv3D/ReadVariableOp�conv3d_3/BiasAdd/ReadVariableOp�conv3d_3/Conv3D/ReadVariableOp�conv3d_4/BiasAdd/ReadVariableOp�conv3d_4/Conv3D/ReadVariableOp�conv3d_5/BiasAdd/ReadVariableOp�conv3d_5/Conv3D/ReadVariableOp�conv3d_6/BiasAdd/ReadVariableOp�conv3d_6/Conv3D/ReadVariableOp�conv3d_7/BiasAdd/ReadVariableOp�conv3d_7/Conv3D/ReadVariableOp�dense/BiasAdd/ReadVariableOp�dense/MatMul/ReadVariableOp�(kernel/Regularizer/Square/ReadVariableOp�*kernel/Regularizer_1/Square/ReadVariableOp�*kernel/Regularizer_2/Square/ReadVariableOp�*kernel/Regularizer_3/Square/ReadVariableOp�*kernel/Regularizer_4/Square/ReadVariableOp�*kernel/Regularizer_5/Square/ReadVariableOp�*kernel/Regularizer_6/Square/ReadVariableOp�*kernel/Regularizer_7/Square/ReadVariableOp�
conv3d/Conv3D/ReadVariableOpReadVariableOp%conv3d_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
conv3d/Conv3DConv3Dinputs$conv3d/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������22*
paddingSAME*
strides	
�
conv3d/BiasAdd/ReadVariableOpReadVariableOp&conv3d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv3d/BiasAddBiasAddconv3d/Conv3D:output:0%conv3d/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������22j
conv3d/ReluReluconv3d/BiasAdd:output:0*
T0*3
_output_shapes!
:���������22�
conv3d_1/Conv3D/ReadVariableOpReadVariableOp'conv3d_1_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
conv3d_1/Conv3DConv3Dconv3d/Relu:activations:0&conv3d_1/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������22*
paddingSAME*
strides	
�
conv3d_1/BiasAdd/ReadVariableOpReadVariableOp(conv3d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv3d_1/BiasAddBiasAddconv3d_1/Conv3D:output:0'conv3d_1/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������22n
conv3d_1/ReluReluconv3d_1/BiasAdd:output:0*
T0*3
_output_shapes!
:���������22�
max_pooling3d/MaxPool3D	MaxPool3Dconv3d_1/Relu:activations:0*
T0*3
_output_shapes!
:���������
*
ksize	
*
paddingVALID*
strides	
�
conv3d_2/Conv3D/ReadVariableOpReadVariableOp'conv3d_2_conv3d_readvariableop_resource**
_output_shapes
: *
dtype0�
conv3d_2/Conv3DConv3D max_pooling3d/MaxPool3D:output:0&conv3d_2/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������
 *
paddingSAME*
strides	
�
conv3d_2/BiasAdd/ReadVariableOpReadVariableOp(conv3d_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
conv3d_2/BiasAddBiasAddconv3d_2/Conv3D:output:0'conv3d_2/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������
 n
conv3d_2/ReluReluconv3d_2/BiasAdd:output:0*
T0*3
_output_shapes!
:���������
 �
conv3d_3/Conv3D/ReadVariableOpReadVariableOp'conv3d_3_conv3d_readvariableop_resource**
_output_shapes
:  *
dtype0�
conv3d_3/Conv3DConv3Dconv3d_2/Relu:activations:0&conv3d_3/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������
 *
paddingSAME*
strides	
�
conv3d_3/BiasAdd/ReadVariableOpReadVariableOp(conv3d_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
conv3d_3/BiasAddBiasAddconv3d_3/Conv3D:output:0'conv3d_3/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������
 n
conv3d_3/ReluReluconv3d_3/BiasAdd:output:0*
T0*3
_output_shapes!
:���������
 �
max_pooling3d_1/MaxPool3D	MaxPool3Dconv3d_3/Relu:activations:0*
T0*3
_output_shapes!
:��������� *
ksize	
*
paddingVALID*
strides	
�
conv3d_4/Conv3D/ReadVariableOpReadVariableOp'conv3d_4_conv3d_readvariableop_resource**
_output_shapes
: @*
dtype0�
conv3d_4/Conv3DConv3D"max_pooling3d_1/MaxPool3D:output:0&conv3d_4/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������@*
paddingSAME*
strides	
�
conv3d_4/BiasAdd/ReadVariableOpReadVariableOp(conv3d_4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
conv3d_4/BiasAddBiasAddconv3d_4/Conv3D:output:0'conv3d_4/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������@n
conv3d_4/ReluReluconv3d_4/BiasAdd:output:0*
T0*3
_output_shapes!
:���������@�
conv3d_5/Conv3D/ReadVariableOpReadVariableOp'conv3d_5_conv3d_readvariableop_resource**
_output_shapes
:@@*
dtype0�
conv3d_5/Conv3DConv3Dconv3d_4/Relu:activations:0&conv3d_5/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������@*
paddingSAME*
strides	
�
conv3d_5/BiasAdd/ReadVariableOpReadVariableOp(conv3d_5_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
conv3d_5/BiasAddBiasAddconv3d_5/Conv3D:output:0'conv3d_5/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������@n
conv3d_5/ReluReluconv3d_5/BiasAdd:output:0*
T0*3
_output_shapes!
:���������@�
max_pooling3d_2/MaxPool3D	MaxPool3Dconv3d_5/Relu:activations:0*
T0*3
_output_shapes!
:���������@*
ksize	
*
paddingVALID*
strides	
�
conv3d_6/Conv3D/ReadVariableOpReadVariableOp'conv3d_6_conv3d_readvariableop_resource*+
_output_shapes
:@�*
dtype0�
conv3d_6/Conv3DConv3D"max_pooling3d_2/MaxPool3D:output:0&conv3d_6/Conv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :����������*
paddingSAME*
strides	
�
conv3d_6/BiasAdd/ReadVariableOpReadVariableOp(conv3d_6_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
conv3d_6/BiasAddBiasAddconv3d_6/Conv3D:output:0'conv3d_6/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :����������o
conv3d_6/ReluReluconv3d_6/BiasAdd:output:0*
T0*4
_output_shapes"
 :�����������
conv3d_7/Conv3D/ReadVariableOpReadVariableOp'conv3d_7_conv3d_readvariableop_resource*,
_output_shapes
:��*
dtype0�
conv3d_7/Conv3DConv3Dconv3d_6/Relu:activations:0&conv3d_7/Conv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :����������*
paddingSAME*
strides	
�
conv3d_7/BiasAdd/ReadVariableOpReadVariableOp(conv3d_7_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
conv3d_7/BiasAddBiasAddconv3d_7/Conv3D:output:0'conv3d_7/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :����������o
conv3d_7/ReluReluconv3d_7/BiasAdd:output:0*
T0*4
_output_shapes"
 :�����������
max_pooling3d_3/MaxPool3D	MaxPool3Dconv3d_7/Relu:activations:0*
T0*4
_output_shapes"
 :����������*
ksize	
*
paddingVALID*
strides	
^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����  �
flatten/ReshapeReshape"max_pooling3d_3/MaxPool3D:output:0flatten/Const:output:0*
T0*(
_output_shapes
:����������	�
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	�	*
dtype0�
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
(kernel/Regularizer/Square/ReadVariableOpReadVariableOp%conv3d_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
kernel/Regularizer/SquareSquare0kernel/Regularizer/Square/ReadVariableOp:value:0*
T0**
_output_shapes
:u
kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
kernel/Regularizer/SumSumkernel/Regularizer/Square:y:0!kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: ]
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;�
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
*kernel/Regularizer_1/Square/ReadVariableOpReadVariableOp'conv3d_1_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
kernel/Regularizer_1/SquareSquare2kernel/Regularizer_1/Square/ReadVariableOp:value:0*
T0**
_output_shapes
:w
kernel/Regularizer_1/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
kernel/Regularizer_1/SumSumkernel/Regularizer_1/Square:y:0#kernel/Regularizer_1/Const:output:0*
T0*
_output_shapes
: _
kernel/Regularizer_1/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;�
kernel/Regularizer_1/mulMul#kernel/Regularizer_1/mul/x:output:0!kernel/Regularizer_1/Sum:output:0*
T0*
_output_shapes
: �
*kernel/Regularizer_2/Square/ReadVariableOpReadVariableOp'conv3d_2_conv3d_readvariableop_resource**
_output_shapes
: *
dtype0�
kernel/Regularizer_2/SquareSquare2kernel/Regularizer_2/Square/ReadVariableOp:value:0*
T0**
_output_shapes
: w
kernel/Regularizer_2/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
kernel/Regularizer_2/SumSumkernel/Regularizer_2/Square:y:0#kernel/Regularizer_2/Const:output:0*
T0*
_output_shapes
: _
kernel/Regularizer_2/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;�
kernel/Regularizer_2/mulMul#kernel/Regularizer_2/mul/x:output:0!kernel/Regularizer_2/Sum:output:0*
T0*
_output_shapes
: �
*kernel/Regularizer_3/Square/ReadVariableOpReadVariableOp'conv3d_3_conv3d_readvariableop_resource**
_output_shapes
:  *
dtype0�
kernel/Regularizer_3/SquareSquare2kernel/Regularizer_3/Square/ReadVariableOp:value:0*
T0**
_output_shapes
:  w
kernel/Regularizer_3/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
kernel/Regularizer_3/SumSumkernel/Regularizer_3/Square:y:0#kernel/Regularizer_3/Const:output:0*
T0*
_output_shapes
: _
kernel/Regularizer_3/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;�
kernel/Regularizer_3/mulMul#kernel/Regularizer_3/mul/x:output:0!kernel/Regularizer_3/Sum:output:0*
T0*
_output_shapes
: �
*kernel/Regularizer_4/Square/ReadVariableOpReadVariableOp'conv3d_4_conv3d_readvariableop_resource**
_output_shapes
: @*
dtype0�
kernel/Regularizer_4/SquareSquare2kernel/Regularizer_4/Square/ReadVariableOp:value:0*
T0**
_output_shapes
: @w
kernel/Regularizer_4/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
kernel/Regularizer_4/SumSumkernel/Regularizer_4/Square:y:0#kernel/Regularizer_4/Const:output:0*
T0*
_output_shapes
: _
kernel/Regularizer_4/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;�
kernel/Regularizer_4/mulMul#kernel/Regularizer_4/mul/x:output:0!kernel/Regularizer_4/Sum:output:0*
T0*
_output_shapes
: �
*kernel/Regularizer_5/Square/ReadVariableOpReadVariableOp'conv3d_5_conv3d_readvariableop_resource**
_output_shapes
:@@*
dtype0�
kernel/Regularizer_5/SquareSquare2kernel/Regularizer_5/Square/ReadVariableOp:value:0*
T0**
_output_shapes
:@@w
kernel/Regularizer_5/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
kernel/Regularizer_5/SumSumkernel/Regularizer_5/Square:y:0#kernel/Regularizer_5/Const:output:0*
T0*
_output_shapes
: _
kernel/Regularizer_5/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;�
kernel/Regularizer_5/mulMul#kernel/Regularizer_5/mul/x:output:0!kernel/Regularizer_5/Sum:output:0*
T0*
_output_shapes
: �
*kernel/Regularizer_6/Square/ReadVariableOpReadVariableOp'conv3d_6_conv3d_readvariableop_resource*+
_output_shapes
:@�*
dtype0�
kernel/Regularizer_6/SquareSquare2kernel/Regularizer_6/Square/ReadVariableOp:value:0*
T0*+
_output_shapes
:@�w
kernel/Regularizer_6/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
kernel/Regularizer_6/SumSumkernel/Regularizer_6/Square:y:0#kernel/Regularizer_6/Const:output:0*
T0*
_output_shapes
: _
kernel/Regularizer_6/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;�
kernel/Regularizer_6/mulMul#kernel/Regularizer_6/mul/x:output:0!kernel/Regularizer_6/Sum:output:0*
T0*
_output_shapes
: �
*kernel/Regularizer_7/Square/ReadVariableOpReadVariableOp'conv3d_7_conv3d_readvariableop_resource*,
_output_shapes
:��*
dtype0�
kernel/Regularizer_7/SquareSquare2kernel/Regularizer_7/Square/ReadVariableOp:value:0*
T0*,
_output_shapes
:��w
kernel/Regularizer_7/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
kernel/Regularizer_7/SumSumkernel/Regularizer_7/Square:y:0#kernel/Regularizer_7/Const:output:0*
T0*
_output_shapes
: _
kernel/Regularizer_7/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;�
kernel/Regularizer_7/mulMul#kernel/Regularizer_7/mul/x:output:0!kernel/Regularizer_7/Sum:output:0*
T0*
_output_shapes
: e
IdentityIdentitydense/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^conv3d/BiasAdd/ReadVariableOp^conv3d/Conv3D/ReadVariableOp ^conv3d_1/BiasAdd/ReadVariableOp^conv3d_1/Conv3D/ReadVariableOp ^conv3d_2/BiasAdd/ReadVariableOp^conv3d_2/Conv3D/ReadVariableOp ^conv3d_3/BiasAdd/ReadVariableOp^conv3d_3/Conv3D/ReadVariableOp ^conv3d_4/BiasAdd/ReadVariableOp^conv3d_4/Conv3D/ReadVariableOp ^conv3d_5/BiasAdd/ReadVariableOp^conv3d_5/Conv3D/ReadVariableOp ^conv3d_6/BiasAdd/ReadVariableOp^conv3d_6/Conv3D/ReadVariableOp ^conv3d_7/BiasAdd/ReadVariableOp^conv3d_7/Conv3D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp)^kernel/Regularizer/Square/ReadVariableOp+^kernel/Regularizer_1/Square/ReadVariableOp+^kernel/Regularizer_2/Square/ReadVariableOp+^kernel/Regularizer_3/Square/ReadVariableOp+^kernel/Regularizer_4/Square/ReadVariableOp+^kernel/Regularizer_5/Square/ReadVariableOp+^kernel/Regularizer_6/Square/ReadVariableOp+^kernel/Regularizer_7/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:���������22: : : : : : : : : : : : : : : : : : 2>
conv3d/BiasAdd/ReadVariableOpconv3d/BiasAdd/ReadVariableOp2<
conv3d/Conv3D/ReadVariableOpconv3d/Conv3D/ReadVariableOp2B
conv3d_1/BiasAdd/ReadVariableOpconv3d_1/BiasAdd/ReadVariableOp2@
conv3d_1/Conv3D/ReadVariableOpconv3d_1/Conv3D/ReadVariableOp2B
conv3d_2/BiasAdd/ReadVariableOpconv3d_2/BiasAdd/ReadVariableOp2@
conv3d_2/Conv3D/ReadVariableOpconv3d_2/Conv3D/ReadVariableOp2B
conv3d_3/BiasAdd/ReadVariableOpconv3d_3/BiasAdd/ReadVariableOp2@
conv3d_3/Conv3D/ReadVariableOpconv3d_3/Conv3D/ReadVariableOp2B
conv3d_4/BiasAdd/ReadVariableOpconv3d_4/BiasAdd/ReadVariableOp2@
conv3d_4/Conv3D/ReadVariableOpconv3d_4/Conv3D/ReadVariableOp2B
conv3d_5/BiasAdd/ReadVariableOpconv3d_5/BiasAdd/ReadVariableOp2@
conv3d_5/Conv3D/ReadVariableOpconv3d_5/Conv3D/ReadVariableOp2B
conv3d_6/BiasAdd/ReadVariableOpconv3d_6/BiasAdd/ReadVariableOp2@
conv3d_6/Conv3D/ReadVariableOpconv3d_6/Conv3D/ReadVariableOp2B
conv3d_7/BiasAdd/ReadVariableOpconv3d_7/BiasAdd/ReadVariableOp2@
conv3d_7/Conv3D/ReadVariableOpconv3d_7/Conv3D/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2T
(kernel/Regularizer/Square/ReadVariableOp(kernel/Regularizer/Square/ReadVariableOp2X
*kernel/Regularizer_1/Square/ReadVariableOp*kernel/Regularizer_1/Square/ReadVariableOp2X
*kernel/Regularizer_2/Square/ReadVariableOp*kernel/Regularizer_2/Square/ReadVariableOp2X
*kernel/Regularizer_3/Square/ReadVariableOp*kernel/Regularizer_3/Square/ReadVariableOp2X
*kernel/Regularizer_4/Square/ReadVariableOp*kernel/Regularizer_4/Square/ReadVariableOp2X
*kernel/Regularizer_5/Square/ReadVariableOp*kernel/Regularizer_5/Square/ReadVariableOp2X
*kernel/Regularizer_6/Square/ReadVariableOp*kernel/Regularizer_6/Square/ReadVariableOp2X
*kernel/Regularizer_7/Square/ReadVariableOp*kernel/Regularizer_7/Square/ReadVariableOp:[ W
3
_output_shapes!
:���������22
 
_user_specified_nameinputs
�
�
C__inference_conv3d_3_layer_call_and_return_conditional_losses_10549

inputs<
conv3d_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�Conv3D/ReadVariableOp�(kernel/Regularizer/Square/ReadVariableOp�
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:  *
dtype0�
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������
 *
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������
 \
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:���������
 �
(kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:  *
dtype0�
kernel/Regularizer/SquareSquare0kernel/Regularizer/Square/ReadVariableOp:value:0*
T0**
_output_shapes
:  u
kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
kernel/Regularizer/SumSumkernel/Regularizer/Square:y:0!kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: ]
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;�
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:���������
 �
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp)^kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������
 : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp2T
(kernel/Regularizer/Square/ReadVariableOp(kernel/Regularizer/Square/ReadVariableOp:[ W
3
_output_shapes!
:���������
 
 
_user_specified_nameinputs
�
�
C__inference_conv3d_6_layer_call_and_return_conditional_losses_11984

inputs=
conv3d_readvariableop_resource:@�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv3D/ReadVariableOp�(kernel/Regularizer/Square/ReadVariableOp�
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource*+
_output_shapes
:@�*
dtype0�
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :����������*
paddingSAME*
strides	
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :����������]
ReluReluBiasAdd:output:0*
T0*4
_output_shapes"
 :�����������
(kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv3d_readvariableop_resource*+
_output_shapes
:@�*
dtype0�
kernel/Regularizer/SquareSquare0kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*+
_output_shapes
:@�u
kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
kernel/Regularizer/SumSumkernel/Regularizer/Square:y:0!kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: ]
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;�
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: n
IdentityIdentityRelu:activations:0^NoOp*
T0*4
_output_shapes"
 :�����������
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp)^kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp2T
(kernel/Regularizer/Square/ReadVariableOp(kernel/Regularizer/Square/ReadVariableOp:[ W
3
_output_shapes!
:���������@
 
_user_specified_nameinputs
�
�
(__inference_conv3d_5_layer_call_fn_11931

inputs%
unknown:@@
	unknown_0:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������@*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *L
fGRE
C__inference_conv3d_5_layer_call_and_return_conditional_losses_10596{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������@: : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:���������@
 
_user_specified_nameinputs
�
�
C__inference_conv3d_2_layer_call_and_return_conditional_losses_11860

inputs<
conv3d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�Conv3D/ReadVariableOp�(kernel/Regularizer/Square/ReadVariableOp�
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
: *
dtype0�
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������
 *
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������
 \
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:���������
 �
(kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
: *
dtype0�
kernel/Regularizer/SquareSquare0kernel/Regularizer/Square/ReadVariableOp:value:0*
T0**
_output_shapes
: u
kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
kernel/Regularizer/SumSumkernel/Regularizer/Square:y:0!kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: ]
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;�
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:���������
 �
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp)^kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp2T
(kernel/Regularizer/Square/ReadVariableOp(kernel/Regularizer/Square/ReadVariableOp:[ W
3
_output_shapes!
:���������

 
_user_specified_nameinputs
�
�
C__inference_conv3d_4_layer_call_and_return_conditional_losses_11922

inputs<
conv3d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�Conv3D/ReadVariableOp�(kernel/Regularizer/Square/ReadVariableOp�
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
: @*
dtype0�
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������@*
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������@\
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:���������@�
(kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
: @*
dtype0�
kernel/Regularizer/SquareSquare0kernel/Regularizer/Square/ReadVariableOp:value:0*
T0**
_output_shapes
: @u
kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
kernel/Regularizer/SumSumkernel/Regularizer/Square:y:0!kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: ]
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;�
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:���������@�
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp)^kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp2T
(kernel/Regularizer/Square/ReadVariableOp(kernel/Regularizer/Square/ReadVariableOp:[ W
3
_output_shapes!
:��������� 
 
_user_specified_nameinputs
�
d
H__inference_max_pooling3d_layer_call_and_return_conditional_losses_11834

inputs
identity�
	MaxPool3D	MaxPool3Dinputs*
T0*W
_output_shapesE
C:A���������������������������������������������*
ksize	
*
paddingVALID*
strides	
�
IdentityIdentityMaxPool3D:output:0*
T0*W
_output_shapesE
C:A���������������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A���������������������������������������������: {
W
_output_shapesE
C:A���������������������������������������������
 
_user_specified_nameinputs
�
�
%__inference_model_layer_call_fn_10762
input_1%
unknown:
	unknown_0:'
	unknown_1:
	unknown_2:'
	unknown_3: 
	unknown_4: '
	unknown_5:  
	unknown_6: '
	unknown_7: @
	unknown_8:@'
	unknown_9:@@

unknown_10:@)

unknown_11:@�

unknown_12:	�*

unknown_13:��

unknown_14:	�

unknown_15:	�	

unknown_16:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*4
_read_only_resource_inputs
	
*4
config_proto$"

CPU

GPU(2*0J 8� *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_10723o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:���������22: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
3
_output_shapes!
:���������22
!
_user_specified_name	input_1
�
�
A__inference_conv3d_layer_call_and_return_conditional_losses_10479

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv3D/ReadVariableOp�(kernel/Regularizer/Square/ReadVariableOp�
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������22*
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������22\
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:���������22�
(kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
kernel/Regularizer/SquareSquare0kernel/Regularizer/Square/ReadVariableOp:value:0*
T0**
_output_shapes
:u
kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
kernel/Regularizer/SumSumkernel/Regularizer/Square:y:0!kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: ]
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;�
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:���������22�
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp)^kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������22: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp2T
(kernel/Regularizer/Square/ReadVariableOp(kernel/Regularizer/Square/ReadVariableOp:[ W
3
_output_shapes!
:���������22
 
_user_specified_nameinputs
�
^
B__inference_flatten_layer_call_and_return_conditional_losses_12031

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"�����  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������	Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������:\ X
4
_output_shapes"
 :����������
 
_user_specified_nameinputs
�
�
%__inference_dense_layer_call_fn_12040

inputs
unknown:	�	
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_10668o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������	: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������	
 
_user_specified_nameinputs
֗
�
@__inference_model_layer_call_and_return_conditional_losses_11652

inputsC
%conv3d_conv3d_readvariableop_resource:4
&conv3d_biasadd_readvariableop_resource:E
'conv3d_1_conv3d_readvariableop_resource:6
(conv3d_1_biasadd_readvariableop_resource:E
'conv3d_2_conv3d_readvariableop_resource: 6
(conv3d_2_biasadd_readvariableop_resource: E
'conv3d_3_conv3d_readvariableop_resource:  6
(conv3d_3_biasadd_readvariableop_resource: E
'conv3d_4_conv3d_readvariableop_resource: @6
(conv3d_4_biasadd_readvariableop_resource:@E
'conv3d_5_conv3d_readvariableop_resource:@@6
(conv3d_5_biasadd_readvariableop_resource:@F
'conv3d_6_conv3d_readvariableop_resource:@�7
(conv3d_6_biasadd_readvariableop_resource:	�G
'conv3d_7_conv3d_readvariableop_resource:��7
(conv3d_7_biasadd_readvariableop_resource:	�7
$dense_matmul_readvariableop_resource:	�	3
%dense_biasadd_readvariableop_resource:
identity��conv3d/BiasAdd/ReadVariableOp�conv3d/Conv3D/ReadVariableOp�conv3d_1/BiasAdd/ReadVariableOp�conv3d_1/Conv3D/ReadVariableOp�conv3d_2/BiasAdd/ReadVariableOp�conv3d_2/Conv3D/ReadVariableOp�conv3d_3/BiasAdd/ReadVariableOp�conv3d_3/Conv3D/ReadVariableOp�conv3d_4/BiasAdd/ReadVariableOp�conv3d_4/Conv3D/ReadVariableOp�conv3d_5/BiasAdd/ReadVariableOp�conv3d_5/Conv3D/ReadVariableOp�conv3d_6/BiasAdd/ReadVariableOp�conv3d_6/Conv3D/ReadVariableOp�conv3d_7/BiasAdd/ReadVariableOp�conv3d_7/Conv3D/ReadVariableOp�dense/BiasAdd/ReadVariableOp�dense/MatMul/ReadVariableOp�(kernel/Regularizer/Square/ReadVariableOp�*kernel/Regularizer_1/Square/ReadVariableOp�*kernel/Regularizer_2/Square/ReadVariableOp�*kernel/Regularizer_3/Square/ReadVariableOp�*kernel/Regularizer_4/Square/ReadVariableOp�*kernel/Regularizer_5/Square/ReadVariableOp�*kernel/Regularizer_6/Square/ReadVariableOp�*kernel/Regularizer_7/Square/ReadVariableOp�
conv3d/Conv3D/ReadVariableOpReadVariableOp%conv3d_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
conv3d/Conv3DConv3Dinputs$conv3d/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������22*
paddingSAME*
strides	
�
conv3d/BiasAdd/ReadVariableOpReadVariableOp&conv3d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv3d/BiasAddBiasAddconv3d/Conv3D:output:0%conv3d/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������22j
conv3d/ReluReluconv3d/BiasAdd:output:0*
T0*3
_output_shapes!
:���������22�
conv3d_1/Conv3D/ReadVariableOpReadVariableOp'conv3d_1_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
conv3d_1/Conv3DConv3Dconv3d/Relu:activations:0&conv3d_1/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������22*
paddingSAME*
strides	
�
conv3d_1/BiasAdd/ReadVariableOpReadVariableOp(conv3d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv3d_1/BiasAddBiasAddconv3d_1/Conv3D:output:0'conv3d_1/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������22n
conv3d_1/ReluReluconv3d_1/BiasAdd:output:0*
T0*3
_output_shapes!
:���������22�
max_pooling3d/MaxPool3D	MaxPool3Dconv3d_1/Relu:activations:0*
T0*3
_output_shapes!
:���������
*
ksize	
*
paddingVALID*
strides	
�
conv3d_2/Conv3D/ReadVariableOpReadVariableOp'conv3d_2_conv3d_readvariableop_resource**
_output_shapes
: *
dtype0�
conv3d_2/Conv3DConv3D max_pooling3d/MaxPool3D:output:0&conv3d_2/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������
 *
paddingSAME*
strides	
�
conv3d_2/BiasAdd/ReadVariableOpReadVariableOp(conv3d_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
conv3d_2/BiasAddBiasAddconv3d_2/Conv3D:output:0'conv3d_2/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������
 n
conv3d_2/ReluReluconv3d_2/BiasAdd:output:0*
T0*3
_output_shapes!
:���������
 �
conv3d_3/Conv3D/ReadVariableOpReadVariableOp'conv3d_3_conv3d_readvariableop_resource**
_output_shapes
:  *
dtype0�
conv3d_3/Conv3DConv3Dconv3d_2/Relu:activations:0&conv3d_3/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������
 *
paddingSAME*
strides	
�
conv3d_3/BiasAdd/ReadVariableOpReadVariableOp(conv3d_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
conv3d_3/BiasAddBiasAddconv3d_3/Conv3D:output:0'conv3d_3/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������
 n
conv3d_3/ReluReluconv3d_3/BiasAdd:output:0*
T0*3
_output_shapes!
:���������
 �
max_pooling3d_1/MaxPool3D	MaxPool3Dconv3d_3/Relu:activations:0*
T0*3
_output_shapes!
:��������� *
ksize	
*
paddingVALID*
strides	
�
conv3d_4/Conv3D/ReadVariableOpReadVariableOp'conv3d_4_conv3d_readvariableop_resource**
_output_shapes
: @*
dtype0�
conv3d_4/Conv3DConv3D"max_pooling3d_1/MaxPool3D:output:0&conv3d_4/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������@*
paddingSAME*
strides	
�
conv3d_4/BiasAdd/ReadVariableOpReadVariableOp(conv3d_4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
conv3d_4/BiasAddBiasAddconv3d_4/Conv3D:output:0'conv3d_4/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������@n
conv3d_4/ReluReluconv3d_4/BiasAdd:output:0*
T0*3
_output_shapes!
:���������@�
conv3d_5/Conv3D/ReadVariableOpReadVariableOp'conv3d_5_conv3d_readvariableop_resource**
_output_shapes
:@@*
dtype0�
conv3d_5/Conv3DConv3Dconv3d_4/Relu:activations:0&conv3d_5/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������@*
paddingSAME*
strides	
�
conv3d_5/BiasAdd/ReadVariableOpReadVariableOp(conv3d_5_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
conv3d_5/BiasAddBiasAddconv3d_5/Conv3D:output:0'conv3d_5/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������@n
conv3d_5/ReluReluconv3d_5/BiasAdd:output:0*
T0*3
_output_shapes!
:���������@�
max_pooling3d_2/MaxPool3D	MaxPool3Dconv3d_5/Relu:activations:0*
T0*3
_output_shapes!
:���������@*
ksize	
*
paddingVALID*
strides	
�
conv3d_6/Conv3D/ReadVariableOpReadVariableOp'conv3d_6_conv3d_readvariableop_resource*+
_output_shapes
:@�*
dtype0�
conv3d_6/Conv3DConv3D"max_pooling3d_2/MaxPool3D:output:0&conv3d_6/Conv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :����������*
paddingSAME*
strides	
�
conv3d_6/BiasAdd/ReadVariableOpReadVariableOp(conv3d_6_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
conv3d_6/BiasAddBiasAddconv3d_6/Conv3D:output:0'conv3d_6/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :����������o
conv3d_6/ReluReluconv3d_6/BiasAdd:output:0*
T0*4
_output_shapes"
 :�����������
conv3d_7/Conv3D/ReadVariableOpReadVariableOp'conv3d_7_conv3d_readvariableop_resource*,
_output_shapes
:��*
dtype0�
conv3d_7/Conv3DConv3Dconv3d_6/Relu:activations:0&conv3d_7/Conv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :����������*
paddingSAME*
strides	
�
conv3d_7/BiasAdd/ReadVariableOpReadVariableOp(conv3d_7_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
conv3d_7/BiasAddBiasAddconv3d_7/Conv3D:output:0'conv3d_7/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :����������o
conv3d_7/ReluReluconv3d_7/BiasAdd:output:0*
T0*4
_output_shapes"
 :�����������
max_pooling3d_3/MaxPool3D	MaxPool3Dconv3d_7/Relu:activations:0*
T0*4
_output_shapes"
 :����������*
ksize	
*
paddingVALID*
strides	
^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����  �
flatten/ReshapeReshape"max_pooling3d_3/MaxPool3D:output:0flatten/Const:output:0*
T0*(
_output_shapes
:����������	�
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	�	*
dtype0�
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
(kernel/Regularizer/Square/ReadVariableOpReadVariableOp%conv3d_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
kernel/Regularizer/SquareSquare0kernel/Regularizer/Square/ReadVariableOp:value:0*
T0**
_output_shapes
:u
kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
kernel/Regularizer/SumSumkernel/Regularizer/Square:y:0!kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: ]
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;�
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
*kernel/Regularizer_1/Square/ReadVariableOpReadVariableOp'conv3d_1_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
kernel/Regularizer_1/SquareSquare2kernel/Regularizer_1/Square/ReadVariableOp:value:0*
T0**
_output_shapes
:w
kernel/Regularizer_1/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
kernel/Regularizer_1/SumSumkernel/Regularizer_1/Square:y:0#kernel/Regularizer_1/Const:output:0*
T0*
_output_shapes
: _
kernel/Regularizer_1/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;�
kernel/Regularizer_1/mulMul#kernel/Regularizer_1/mul/x:output:0!kernel/Regularizer_1/Sum:output:0*
T0*
_output_shapes
: �
*kernel/Regularizer_2/Square/ReadVariableOpReadVariableOp'conv3d_2_conv3d_readvariableop_resource**
_output_shapes
: *
dtype0�
kernel/Regularizer_2/SquareSquare2kernel/Regularizer_2/Square/ReadVariableOp:value:0*
T0**
_output_shapes
: w
kernel/Regularizer_2/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
kernel/Regularizer_2/SumSumkernel/Regularizer_2/Square:y:0#kernel/Regularizer_2/Const:output:0*
T0*
_output_shapes
: _
kernel/Regularizer_2/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;�
kernel/Regularizer_2/mulMul#kernel/Regularizer_2/mul/x:output:0!kernel/Regularizer_2/Sum:output:0*
T0*
_output_shapes
: �
*kernel/Regularizer_3/Square/ReadVariableOpReadVariableOp'conv3d_3_conv3d_readvariableop_resource**
_output_shapes
:  *
dtype0�
kernel/Regularizer_3/SquareSquare2kernel/Regularizer_3/Square/ReadVariableOp:value:0*
T0**
_output_shapes
:  w
kernel/Regularizer_3/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
kernel/Regularizer_3/SumSumkernel/Regularizer_3/Square:y:0#kernel/Regularizer_3/Const:output:0*
T0*
_output_shapes
: _
kernel/Regularizer_3/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;�
kernel/Regularizer_3/mulMul#kernel/Regularizer_3/mul/x:output:0!kernel/Regularizer_3/Sum:output:0*
T0*
_output_shapes
: �
*kernel/Regularizer_4/Square/ReadVariableOpReadVariableOp'conv3d_4_conv3d_readvariableop_resource**
_output_shapes
: @*
dtype0�
kernel/Regularizer_4/SquareSquare2kernel/Regularizer_4/Square/ReadVariableOp:value:0*
T0**
_output_shapes
: @w
kernel/Regularizer_4/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
kernel/Regularizer_4/SumSumkernel/Regularizer_4/Square:y:0#kernel/Regularizer_4/Const:output:0*
T0*
_output_shapes
: _
kernel/Regularizer_4/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;�
kernel/Regularizer_4/mulMul#kernel/Regularizer_4/mul/x:output:0!kernel/Regularizer_4/Sum:output:0*
T0*
_output_shapes
: �
*kernel/Regularizer_5/Square/ReadVariableOpReadVariableOp'conv3d_5_conv3d_readvariableop_resource**
_output_shapes
:@@*
dtype0�
kernel/Regularizer_5/SquareSquare2kernel/Regularizer_5/Square/ReadVariableOp:value:0*
T0**
_output_shapes
:@@w
kernel/Regularizer_5/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
kernel/Regularizer_5/SumSumkernel/Regularizer_5/Square:y:0#kernel/Regularizer_5/Const:output:0*
T0*
_output_shapes
: _
kernel/Regularizer_5/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;�
kernel/Regularizer_5/mulMul#kernel/Regularizer_5/mul/x:output:0!kernel/Regularizer_5/Sum:output:0*
T0*
_output_shapes
: �
*kernel/Regularizer_6/Square/ReadVariableOpReadVariableOp'conv3d_6_conv3d_readvariableop_resource*+
_output_shapes
:@�*
dtype0�
kernel/Regularizer_6/SquareSquare2kernel/Regularizer_6/Square/ReadVariableOp:value:0*
T0*+
_output_shapes
:@�w
kernel/Regularizer_6/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
kernel/Regularizer_6/SumSumkernel/Regularizer_6/Square:y:0#kernel/Regularizer_6/Const:output:0*
T0*
_output_shapes
: _
kernel/Regularizer_6/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;�
kernel/Regularizer_6/mulMul#kernel/Regularizer_6/mul/x:output:0!kernel/Regularizer_6/Sum:output:0*
T0*
_output_shapes
: �
*kernel/Regularizer_7/Square/ReadVariableOpReadVariableOp'conv3d_7_conv3d_readvariableop_resource*,
_output_shapes
:��*
dtype0�
kernel/Regularizer_7/SquareSquare2kernel/Regularizer_7/Square/ReadVariableOp:value:0*
T0*,
_output_shapes
:��w
kernel/Regularizer_7/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
kernel/Regularizer_7/SumSumkernel/Regularizer_7/Square:y:0#kernel/Regularizer_7/Const:output:0*
T0*
_output_shapes
: _
kernel/Regularizer_7/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;�
kernel/Regularizer_7/mulMul#kernel/Regularizer_7/mul/x:output:0!kernel/Regularizer_7/Sum:output:0*
T0*
_output_shapes
: e
IdentityIdentitydense/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^conv3d/BiasAdd/ReadVariableOp^conv3d/Conv3D/ReadVariableOp ^conv3d_1/BiasAdd/ReadVariableOp^conv3d_1/Conv3D/ReadVariableOp ^conv3d_2/BiasAdd/ReadVariableOp^conv3d_2/Conv3D/ReadVariableOp ^conv3d_3/BiasAdd/ReadVariableOp^conv3d_3/Conv3D/ReadVariableOp ^conv3d_4/BiasAdd/ReadVariableOp^conv3d_4/Conv3D/ReadVariableOp ^conv3d_5/BiasAdd/ReadVariableOp^conv3d_5/Conv3D/ReadVariableOp ^conv3d_6/BiasAdd/ReadVariableOp^conv3d_6/Conv3D/ReadVariableOp ^conv3d_7/BiasAdd/ReadVariableOp^conv3d_7/Conv3D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp)^kernel/Regularizer/Square/ReadVariableOp+^kernel/Regularizer_1/Square/ReadVariableOp+^kernel/Regularizer_2/Square/ReadVariableOp+^kernel/Regularizer_3/Square/ReadVariableOp+^kernel/Regularizer_4/Square/ReadVariableOp+^kernel/Regularizer_5/Square/ReadVariableOp+^kernel/Regularizer_6/Square/ReadVariableOp+^kernel/Regularizer_7/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:���������22: : : : : : : : : : : : : : : : : : 2>
conv3d/BiasAdd/ReadVariableOpconv3d/BiasAdd/ReadVariableOp2<
conv3d/Conv3D/ReadVariableOpconv3d/Conv3D/ReadVariableOp2B
conv3d_1/BiasAdd/ReadVariableOpconv3d_1/BiasAdd/ReadVariableOp2@
conv3d_1/Conv3D/ReadVariableOpconv3d_1/Conv3D/ReadVariableOp2B
conv3d_2/BiasAdd/ReadVariableOpconv3d_2/BiasAdd/ReadVariableOp2@
conv3d_2/Conv3D/ReadVariableOpconv3d_2/Conv3D/ReadVariableOp2B
conv3d_3/BiasAdd/ReadVariableOpconv3d_3/BiasAdd/ReadVariableOp2@
conv3d_3/Conv3D/ReadVariableOpconv3d_3/Conv3D/ReadVariableOp2B
conv3d_4/BiasAdd/ReadVariableOpconv3d_4/BiasAdd/ReadVariableOp2@
conv3d_4/Conv3D/ReadVariableOpconv3d_4/Conv3D/ReadVariableOp2B
conv3d_5/BiasAdd/ReadVariableOpconv3d_5/BiasAdd/ReadVariableOp2@
conv3d_5/Conv3D/ReadVariableOpconv3d_5/Conv3D/ReadVariableOp2B
conv3d_6/BiasAdd/ReadVariableOpconv3d_6/BiasAdd/ReadVariableOp2@
conv3d_6/Conv3D/ReadVariableOpconv3d_6/Conv3D/ReadVariableOp2B
conv3d_7/BiasAdd/ReadVariableOpconv3d_7/BiasAdd/ReadVariableOp2@
conv3d_7/Conv3D/ReadVariableOpconv3d_7/Conv3D/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2T
(kernel/Regularizer/Square/ReadVariableOp(kernel/Regularizer/Square/ReadVariableOp2X
*kernel/Regularizer_1/Square/ReadVariableOp*kernel/Regularizer_1/Square/ReadVariableOp2X
*kernel/Regularizer_2/Square/ReadVariableOp*kernel/Regularizer_2/Square/ReadVariableOp2X
*kernel/Regularizer_3/Square/ReadVariableOp*kernel/Regularizer_3/Square/ReadVariableOp2X
*kernel/Regularizer_4/Square/ReadVariableOp*kernel/Regularizer_4/Square/ReadVariableOp2X
*kernel/Regularizer_5/Square/ReadVariableOp*kernel/Regularizer_5/Square/ReadVariableOp2X
*kernel/Regularizer_6/Square/ReadVariableOp*kernel/Regularizer_6/Square/ReadVariableOp2X
*kernel/Regularizer_7/Square/ReadVariableOp*kernel/Regularizer_7/Square/ReadVariableOp:[ W
3
_output_shapes!
:���������22
 
_user_specified_nameinputs
�

�
__inference_loss_fn_4_12105O
1kernel_regularizer_square_readvariableop_resource: @
identity��(kernel/Regularizer/Square/ReadVariableOp�
(kernel/Regularizer/Square/ReadVariableOpReadVariableOp1kernel_regularizer_square_readvariableop_resource**
_output_shapes
: @*
dtype0�
kernel/Regularizer/SquareSquare0kernel/Regularizer/Square/ReadVariableOp:value:0*
T0**
_output_shapes
: @u
kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
kernel/Regularizer/SumSumkernel/Regularizer/Square:y:0!kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: ]
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;�
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: X
IdentityIdentitykernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: q
NoOpNoOp)^kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2T
(kernel/Regularizer/Square/ReadVariableOp(kernel/Regularizer/Square/ReadVariableOp
�
�
(__inference_conv3d_2_layer_call_fn_11843

inputs%
unknown: 
	unknown_0: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������
 *$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *L
fGRE
C__inference_conv3d_2_layer_call_and_return_conditional_losses_10526{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:���������
 `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������
: : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:���������

 
_user_specified_nameinputs
�
�
%__inference_model_layer_call_fn_11083
input_1%
unknown:
	unknown_0:'
	unknown_1:
	unknown_2:'
	unknown_3: 
	unknown_4: '
	unknown_5:  
	unknown_6: '
	unknown_7: @
	unknown_8:@'
	unknown_9:@@

unknown_10:@)

unknown_11:@�

unknown_12:	�*

unknown_13:��

unknown_14:	�

unknown_15:	�	

unknown_16:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*4
_read_only_resource_inputs
	
*4
config_proto$"

CPU

GPU(2*0J 8� *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_11003o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:���������22: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
3
_output_shapes!
:���������22
!
_user_specified_name	input_1
�
�
C__inference_conv3d_3_layer_call_and_return_conditional_losses_11886

inputs<
conv3d_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�Conv3D/ReadVariableOp�(kernel/Regularizer/Square/ReadVariableOp�
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:  *
dtype0�
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������
 *
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������
 \
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:���������
 �
(kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:  *
dtype0�
kernel/Regularizer/SquareSquare0kernel/Regularizer/Square/ReadVariableOp:value:0*
T0**
_output_shapes
:  u
kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
kernel/Regularizer/SumSumkernel/Regularizer/Square:y:0!kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: ]
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;�
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:���������
 �
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp)^kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������
 : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp2T
(kernel/Regularizer/Square/ReadVariableOp(kernel/Regularizer/Square/ReadVariableOp:[ W
3
_output_shapes!
:���������
 
 
_user_specified_nameinputs
�
K
/__inference_max_pooling3d_2_layer_call_fn_11953

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *W
_output_shapesE
C:A���������������������������������������������* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8� *S
fNRL
J__inference_max_pooling3d_2_layer_call_and_return_conditional_losses_10440�
IdentityIdentityPartitionedCall:output:0*
T0*W
_output_shapesE
C:A���������������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A���������������������������������������������: {
W
_output_shapesE
C:A���������������������������������������������
 
_user_specified_nameinputs
�
�
(__inference_conv3d_7_layer_call_fn_11993

inputs'
unknown:��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :����������*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *L
fGRE
C__inference_conv3d_7_layer_call_and_return_conditional_losses_10643|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :����������
 
_user_specified_nameinputs
�z
�
@__inference_model_layer_call_and_return_conditional_losses_11003

inputs*
conv3d_10904:
conv3d_10906:,
conv3d_1_10909:
conv3d_1_10911:,
conv3d_2_10915: 
conv3d_2_10917: ,
conv3d_3_10920:  
conv3d_3_10922: ,
conv3d_4_10926: @
conv3d_4_10928:@,
conv3d_5_10931:@@
conv3d_5_10933:@-
conv3d_6_10937:@�
conv3d_6_10939:	�.
conv3d_7_10942:��
conv3d_7_10944:	�
dense_10949:	�	
dense_10951:
identity��conv3d/StatefulPartitionedCall� conv3d_1/StatefulPartitionedCall� conv3d_2/StatefulPartitionedCall� conv3d_3/StatefulPartitionedCall� conv3d_4/StatefulPartitionedCall� conv3d_5/StatefulPartitionedCall� conv3d_6/StatefulPartitionedCall� conv3d_7/StatefulPartitionedCall�dense/StatefulPartitionedCall�(kernel/Regularizer/Square/ReadVariableOp�*kernel/Regularizer_1/Square/ReadVariableOp�*kernel/Regularizer_2/Square/ReadVariableOp�*kernel/Regularizer_3/Square/ReadVariableOp�*kernel/Regularizer_4/Square/ReadVariableOp�*kernel/Regularizer_5/Square/ReadVariableOp�*kernel/Regularizer_6/Square/ReadVariableOp�*kernel/Regularizer_7/Square/ReadVariableOp�
conv3d/StatefulPartitionedCallStatefulPartitionedCallinputsconv3d_10904conv3d_10906*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������22*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *J
fERC
A__inference_conv3d_layer_call_and_return_conditional_losses_10479�
 conv3d_1/StatefulPartitionedCallStatefulPartitionedCall'conv3d/StatefulPartitionedCall:output:0conv3d_1_10909conv3d_1_10911*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������22*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *L
fGRE
C__inference_conv3d_1_layer_call_and_return_conditional_losses_10502�
max_pooling3d/PartitionedCallPartitionedCall)conv3d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������
* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8� *Q
fLRJ
H__inference_max_pooling3d_layer_call_and_return_conditional_losses_10416�
 conv3d_2/StatefulPartitionedCallStatefulPartitionedCall&max_pooling3d/PartitionedCall:output:0conv3d_2_10915conv3d_2_10917*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������
 *$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *L
fGRE
C__inference_conv3d_2_layer_call_and_return_conditional_losses_10526�
 conv3d_3/StatefulPartitionedCallStatefulPartitionedCall)conv3d_2/StatefulPartitionedCall:output:0conv3d_3_10920conv3d_3_10922*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������
 *$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *L
fGRE
C__inference_conv3d_3_layer_call_and_return_conditional_losses_10549�
max_pooling3d_1/PartitionedCallPartitionedCall)conv3d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:��������� * 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8� *S
fNRL
J__inference_max_pooling3d_1_layer_call_and_return_conditional_losses_10428�
 conv3d_4/StatefulPartitionedCallStatefulPartitionedCall(max_pooling3d_1/PartitionedCall:output:0conv3d_4_10926conv3d_4_10928*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������@*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *L
fGRE
C__inference_conv3d_4_layer_call_and_return_conditional_losses_10573�
 conv3d_5/StatefulPartitionedCallStatefulPartitionedCall)conv3d_4/StatefulPartitionedCall:output:0conv3d_5_10931conv3d_5_10933*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������@*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *L
fGRE
C__inference_conv3d_5_layer_call_and_return_conditional_losses_10596�
max_pooling3d_2/PartitionedCallPartitionedCall)conv3d_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������@* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8� *S
fNRL
J__inference_max_pooling3d_2_layer_call_and_return_conditional_losses_10440�
 conv3d_6/StatefulPartitionedCallStatefulPartitionedCall(max_pooling3d_2/PartitionedCall:output:0conv3d_6_10937conv3d_6_10939*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :����������*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *L
fGRE
C__inference_conv3d_6_layer_call_and_return_conditional_losses_10620�
 conv3d_7/StatefulPartitionedCallStatefulPartitionedCall)conv3d_6/StatefulPartitionedCall:output:0conv3d_7_10942conv3d_7_10944*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :����������*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *L
fGRE
C__inference_conv3d_7_layer_call_and_return_conditional_losses_10643�
max_pooling3d_3/PartitionedCallPartitionedCall)conv3d_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :����������* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8� *S
fNRL
J__inference_max_pooling3d_3_layer_call_and_return_conditional_losses_10452�
flatten/PartitionedCallPartitionedCall(max_pooling3d_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������	* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8� *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_10656�
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_10949dense_10951*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_10668�
(kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv3d_10904**
_output_shapes
:*
dtype0�
kernel/Regularizer/SquareSquare0kernel/Regularizer/Square/ReadVariableOp:value:0*
T0**
_output_shapes
:u
kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
kernel/Regularizer/SumSumkernel/Regularizer/Square:y:0!kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: ]
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;�
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
*kernel/Regularizer_1/Square/ReadVariableOpReadVariableOpconv3d_1_10909**
_output_shapes
:*
dtype0�
kernel/Regularizer_1/SquareSquare2kernel/Regularizer_1/Square/ReadVariableOp:value:0*
T0**
_output_shapes
:w
kernel/Regularizer_1/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
kernel/Regularizer_1/SumSumkernel/Regularizer_1/Square:y:0#kernel/Regularizer_1/Const:output:0*
T0*
_output_shapes
: _
kernel/Regularizer_1/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;�
kernel/Regularizer_1/mulMul#kernel/Regularizer_1/mul/x:output:0!kernel/Regularizer_1/Sum:output:0*
T0*
_output_shapes
: �
*kernel/Regularizer_2/Square/ReadVariableOpReadVariableOpconv3d_2_10915**
_output_shapes
: *
dtype0�
kernel/Regularizer_2/SquareSquare2kernel/Regularizer_2/Square/ReadVariableOp:value:0*
T0**
_output_shapes
: w
kernel/Regularizer_2/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
kernel/Regularizer_2/SumSumkernel/Regularizer_2/Square:y:0#kernel/Regularizer_2/Const:output:0*
T0*
_output_shapes
: _
kernel/Regularizer_2/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;�
kernel/Regularizer_2/mulMul#kernel/Regularizer_2/mul/x:output:0!kernel/Regularizer_2/Sum:output:0*
T0*
_output_shapes
: �
*kernel/Regularizer_3/Square/ReadVariableOpReadVariableOpconv3d_3_10920**
_output_shapes
:  *
dtype0�
kernel/Regularizer_3/SquareSquare2kernel/Regularizer_3/Square/ReadVariableOp:value:0*
T0**
_output_shapes
:  w
kernel/Regularizer_3/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
kernel/Regularizer_3/SumSumkernel/Regularizer_3/Square:y:0#kernel/Regularizer_3/Const:output:0*
T0*
_output_shapes
: _
kernel/Regularizer_3/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;�
kernel/Regularizer_3/mulMul#kernel/Regularizer_3/mul/x:output:0!kernel/Regularizer_3/Sum:output:0*
T0*
_output_shapes
: �
*kernel/Regularizer_4/Square/ReadVariableOpReadVariableOpconv3d_4_10926**
_output_shapes
: @*
dtype0�
kernel/Regularizer_4/SquareSquare2kernel/Regularizer_4/Square/ReadVariableOp:value:0*
T0**
_output_shapes
: @w
kernel/Regularizer_4/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
kernel/Regularizer_4/SumSumkernel/Regularizer_4/Square:y:0#kernel/Regularizer_4/Const:output:0*
T0*
_output_shapes
: _
kernel/Regularizer_4/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;�
kernel/Regularizer_4/mulMul#kernel/Regularizer_4/mul/x:output:0!kernel/Regularizer_4/Sum:output:0*
T0*
_output_shapes
: �
*kernel/Regularizer_5/Square/ReadVariableOpReadVariableOpconv3d_5_10931**
_output_shapes
:@@*
dtype0�
kernel/Regularizer_5/SquareSquare2kernel/Regularizer_5/Square/ReadVariableOp:value:0*
T0**
_output_shapes
:@@w
kernel/Regularizer_5/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
kernel/Regularizer_5/SumSumkernel/Regularizer_5/Square:y:0#kernel/Regularizer_5/Const:output:0*
T0*
_output_shapes
: _
kernel/Regularizer_5/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;�
kernel/Regularizer_5/mulMul#kernel/Regularizer_5/mul/x:output:0!kernel/Regularizer_5/Sum:output:0*
T0*
_output_shapes
: �
*kernel/Regularizer_6/Square/ReadVariableOpReadVariableOpconv3d_6_10937*+
_output_shapes
:@�*
dtype0�
kernel/Regularizer_6/SquareSquare2kernel/Regularizer_6/Square/ReadVariableOp:value:0*
T0*+
_output_shapes
:@�w
kernel/Regularizer_6/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
kernel/Regularizer_6/SumSumkernel/Regularizer_6/Square:y:0#kernel/Regularizer_6/Const:output:0*
T0*
_output_shapes
: _
kernel/Regularizer_6/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;�
kernel/Regularizer_6/mulMul#kernel/Regularizer_6/mul/x:output:0!kernel/Regularizer_6/Sum:output:0*
T0*
_output_shapes
: �
*kernel/Regularizer_7/Square/ReadVariableOpReadVariableOpconv3d_7_10942*,
_output_shapes
:��*
dtype0�
kernel/Regularizer_7/SquareSquare2kernel/Regularizer_7/Square/ReadVariableOp:value:0*
T0*,
_output_shapes
:��w
kernel/Regularizer_7/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
kernel/Regularizer_7/SumSumkernel/Regularizer_7/Square:y:0#kernel/Regularizer_7/Const:output:0*
T0*
_output_shapes
: _
kernel/Regularizer_7/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;�
kernel/Regularizer_7/mulMul#kernel/Regularizer_7/mul/x:output:0!kernel/Regularizer_7/Sum:output:0*
T0*
_output_shapes
: u
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^conv3d/StatefulPartitionedCall!^conv3d_1/StatefulPartitionedCall!^conv3d_2/StatefulPartitionedCall!^conv3d_3/StatefulPartitionedCall!^conv3d_4/StatefulPartitionedCall!^conv3d_5/StatefulPartitionedCall!^conv3d_6/StatefulPartitionedCall!^conv3d_7/StatefulPartitionedCall^dense/StatefulPartitionedCall)^kernel/Regularizer/Square/ReadVariableOp+^kernel/Regularizer_1/Square/ReadVariableOp+^kernel/Regularizer_2/Square/ReadVariableOp+^kernel/Regularizer_3/Square/ReadVariableOp+^kernel/Regularizer_4/Square/ReadVariableOp+^kernel/Regularizer_5/Square/ReadVariableOp+^kernel/Regularizer_6/Square/ReadVariableOp+^kernel/Regularizer_7/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:���������22: : : : : : : : : : : : : : : : : : 2@
conv3d/StatefulPartitionedCallconv3d/StatefulPartitionedCall2D
 conv3d_1/StatefulPartitionedCall conv3d_1/StatefulPartitionedCall2D
 conv3d_2/StatefulPartitionedCall conv3d_2/StatefulPartitionedCall2D
 conv3d_3/StatefulPartitionedCall conv3d_3/StatefulPartitionedCall2D
 conv3d_4/StatefulPartitionedCall conv3d_4/StatefulPartitionedCall2D
 conv3d_5/StatefulPartitionedCall conv3d_5/StatefulPartitionedCall2D
 conv3d_6/StatefulPartitionedCall conv3d_6/StatefulPartitionedCall2D
 conv3d_7/StatefulPartitionedCall conv3d_7/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2T
(kernel/Regularizer/Square/ReadVariableOp(kernel/Regularizer/Square/ReadVariableOp2X
*kernel/Regularizer_1/Square/ReadVariableOp*kernel/Regularizer_1/Square/ReadVariableOp2X
*kernel/Regularizer_2/Square/ReadVariableOp*kernel/Regularizer_2/Square/ReadVariableOp2X
*kernel/Regularizer_3/Square/ReadVariableOp*kernel/Regularizer_3/Square/ReadVariableOp2X
*kernel/Regularizer_4/Square/ReadVariableOp*kernel/Regularizer_4/Square/ReadVariableOp2X
*kernel/Regularizer_5/Square/ReadVariableOp*kernel/Regularizer_5/Square/ReadVariableOp2X
*kernel/Regularizer_6/Square/ReadVariableOp*kernel/Regularizer_6/Square/ReadVariableOp2X
*kernel/Regularizer_7/Square/ReadVariableOp*kernel/Regularizer_7/Square/ReadVariableOp:[ W
3
_output_shapes!
:���������22
 
_user_specified_nameinputs
�	
�
@__inference_dense_layer_call_and_return_conditional_losses_12050

inputs1
matmul_readvariableop_resource:	�	-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�	*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������	
 
_user_specified_nameinputs
�
A
__inference_loss_fn_11350

y_true

y_pred
identityC
subSuby_truey_pred*
T0*
_output_shapes

: B
SquareSquaresub:z:0*
T0*
_output_shapes

: V
ConstConst*
_output_shapes
:*
dtype0*
valueB"       I
MeanMean
Square:y:0Const:output:0*
T0*
_output_shapes
: D
IdentityIdentityMean:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
: : :F B

_output_shapes

: 
 
_user_specified_namey_true:FB

_output_shapes

: 
 
_user_specified_namey_pred
�
A
__inference_loss_fn_11359

y_true

y_pred
identityL
subSuby_truey_pred*
T0*'
_output_shapes
:���������K
SquareSquaresub:z:0*
T0*'
_output_shapes
:���������V
ConstConst*
_output_shapes
:*
dtype0*
valueB"       I
MeanMean
Square:y:0Const:output:0*
T0*
_output_shapes
: D
IdentityIdentityMean:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:���������:���������:O K
'
_output_shapes
:���������
 
_user_specified_namey_true:OK
'
_output_shapes
:���������
 
_user_specified_namey_pred
�
f
J__inference_max_pooling3d_3_layer_call_and_return_conditional_losses_12020

inputs
identity�
	MaxPool3D	MaxPool3Dinputs*
T0*W
_output_shapesE
C:A���������������������������������������������*
ksize	
*
paddingVALID*
strides	
�
IdentityIdentityMaxPool3D:output:0*
T0*W
_output_shapesE
C:A���������������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A���������������������������������������������: {
W
_output_shapesE
C:A���������������������������������������������
 
_user_specified_nameinputs
�z
�
@__inference_model_layer_call_and_return_conditional_losses_10723

inputs*
conv3d_10480:
conv3d_10482:,
conv3d_1_10503:
conv3d_1_10505:,
conv3d_2_10527: 
conv3d_2_10529: ,
conv3d_3_10550:  
conv3d_3_10552: ,
conv3d_4_10574: @
conv3d_4_10576:@,
conv3d_5_10597:@@
conv3d_5_10599:@-
conv3d_6_10621:@�
conv3d_6_10623:	�.
conv3d_7_10644:��
conv3d_7_10646:	�
dense_10669:	�	
dense_10671:
identity��conv3d/StatefulPartitionedCall� conv3d_1/StatefulPartitionedCall� conv3d_2/StatefulPartitionedCall� conv3d_3/StatefulPartitionedCall� conv3d_4/StatefulPartitionedCall� conv3d_5/StatefulPartitionedCall� conv3d_6/StatefulPartitionedCall� conv3d_7/StatefulPartitionedCall�dense/StatefulPartitionedCall�(kernel/Regularizer/Square/ReadVariableOp�*kernel/Regularizer_1/Square/ReadVariableOp�*kernel/Regularizer_2/Square/ReadVariableOp�*kernel/Regularizer_3/Square/ReadVariableOp�*kernel/Regularizer_4/Square/ReadVariableOp�*kernel/Regularizer_5/Square/ReadVariableOp�*kernel/Regularizer_6/Square/ReadVariableOp�*kernel/Regularizer_7/Square/ReadVariableOp�
conv3d/StatefulPartitionedCallStatefulPartitionedCallinputsconv3d_10480conv3d_10482*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������22*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *J
fERC
A__inference_conv3d_layer_call_and_return_conditional_losses_10479�
 conv3d_1/StatefulPartitionedCallStatefulPartitionedCall'conv3d/StatefulPartitionedCall:output:0conv3d_1_10503conv3d_1_10505*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������22*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *L
fGRE
C__inference_conv3d_1_layer_call_and_return_conditional_losses_10502�
max_pooling3d/PartitionedCallPartitionedCall)conv3d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������
* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8� *Q
fLRJ
H__inference_max_pooling3d_layer_call_and_return_conditional_losses_10416�
 conv3d_2/StatefulPartitionedCallStatefulPartitionedCall&max_pooling3d/PartitionedCall:output:0conv3d_2_10527conv3d_2_10529*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������
 *$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *L
fGRE
C__inference_conv3d_2_layer_call_and_return_conditional_losses_10526�
 conv3d_3/StatefulPartitionedCallStatefulPartitionedCall)conv3d_2/StatefulPartitionedCall:output:0conv3d_3_10550conv3d_3_10552*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������
 *$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *L
fGRE
C__inference_conv3d_3_layer_call_and_return_conditional_losses_10549�
max_pooling3d_1/PartitionedCallPartitionedCall)conv3d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:��������� * 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8� *S
fNRL
J__inference_max_pooling3d_1_layer_call_and_return_conditional_losses_10428�
 conv3d_4/StatefulPartitionedCallStatefulPartitionedCall(max_pooling3d_1/PartitionedCall:output:0conv3d_4_10574conv3d_4_10576*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������@*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *L
fGRE
C__inference_conv3d_4_layer_call_and_return_conditional_losses_10573�
 conv3d_5/StatefulPartitionedCallStatefulPartitionedCall)conv3d_4/StatefulPartitionedCall:output:0conv3d_5_10597conv3d_5_10599*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������@*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *L
fGRE
C__inference_conv3d_5_layer_call_and_return_conditional_losses_10596�
max_pooling3d_2/PartitionedCallPartitionedCall)conv3d_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������@* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8� *S
fNRL
J__inference_max_pooling3d_2_layer_call_and_return_conditional_losses_10440�
 conv3d_6/StatefulPartitionedCallStatefulPartitionedCall(max_pooling3d_2/PartitionedCall:output:0conv3d_6_10621conv3d_6_10623*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :����������*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *L
fGRE
C__inference_conv3d_6_layer_call_and_return_conditional_losses_10620�
 conv3d_7/StatefulPartitionedCallStatefulPartitionedCall)conv3d_6/StatefulPartitionedCall:output:0conv3d_7_10644conv3d_7_10646*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :����������*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *L
fGRE
C__inference_conv3d_7_layer_call_and_return_conditional_losses_10643�
max_pooling3d_3/PartitionedCallPartitionedCall)conv3d_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :����������* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8� *S
fNRL
J__inference_max_pooling3d_3_layer_call_and_return_conditional_losses_10452�
flatten/PartitionedCallPartitionedCall(max_pooling3d_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������	* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8� *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_10656�
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_10669dense_10671*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_10668�
(kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv3d_10480**
_output_shapes
:*
dtype0�
kernel/Regularizer/SquareSquare0kernel/Regularizer/Square/ReadVariableOp:value:0*
T0**
_output_shapes
:u
kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
kernel/Regularizer/SumSumkernel/Regularizer/Square:y:0!kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: ]
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;�
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
*kernel/Regularizer_1/Square/ReadVariableOpReadVariableOpconv3d_1_10503**
_output_shapes
:*
dtype0�
kernel/Regularizer_1/SquareSquare2kernel/Regularizer_1/Square/ReadVariableOp:value:0*
T0**
_output_shapes
:w
kernel/Regularizer_1/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
kernel/Regularizer_1/SumSumkernel/Regularizer_1/Square:y:0#kernel/Regularizer_1/Const:output:0*
T0*
_output_shapes
: _
kernel/Regularizer_1/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;�
kernel/Regularizer_1/mulMul#kernel/Regularizer_1/mul/x:output:0!kernel/Regularizer_1/Sum:output:0*
T0*
_output_shapes
: �
*kernel/Regularizer_2/Square/ReadVariableOpReadVariableOpconv3d_2_10527**
_output_shapes
: *
dtype0�
kernel/Regularizer_2/SquareSquare2kernel/Regularizer_2/Square/ReadVariableOp:value:0*
T0**
_output_shapes
: w
kernel/Regularizer_2/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
kernel/Regularizer_2/SumSumkernel/Regularizer_2/Square:y:0#kernel/Regularizer_2/Const:output:0*
T0*
_output_shapes
: _
kernel/Regularizer_2/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;�
kernel/Regularizer_2/mulMul#kernel/Regularizer_2/mul/x:output:0!kernel/Regularizer_2/Sum:output:0*
T0*
_output_shapes
: �
*kernel/Regularizer_3/Square/ReadVariableOpReadVariableOpconv3d_3_10550**
_output_shapes
:  *
dtype0�
kernel/Regularizer_3/SquareSquare2kernel/Regularizer_3/Square/ReadVariableOp:value:0*
T0**
_output_shapes
:  w
kernel/Regularizer_3/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
kernel/Regularizer_3/SumSumkernel/Regularizer_3/Square:y:0#kernel/Regularizer_3/Const:output:0*
T0*
_output_shapes
: _
kernel/Regularizer_3/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;�
kernel/Regularizer_3/mulMul#kernel/Regularizer_3/mul/x:output:0!kernel/Regularizer_3/Sum:output:0*
T0*
_output_shapes
: �
*kernel/Regularizer_4/Square/ReadVariableOpReadVariableOpconv3d_4_10574**
_output_shapes
: @*
dtype0�
kernel/Regularizer_4/SquareSquare2kernel/Regularizer_4/Square/ReadVariableOp:value:0*
T0**
_output_shapes
: @w
kernel/Regularizer_4/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
kernel/Regularizer_4/SumSumkernel/Regularizer_4/Square:y:0#kernel/Regularizer_4/Const:output:0*
T0*
_output_shapes
: _
kernel/Regularizer_4/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;�
kernel/Regularizer_4/mulMul#kernel/Regularizer_4/mul/x:output:0!kernel/Regularizer_4/Sum:output:0*
T0*
_output_shapes
: �
*kernel/Regularizer_5/Square/ReadVariableOpReadVariableOpconv3d_5_10597**
_output_shapes
:@@*
dtype0�
kernel/Regularizer_5/SquareSquare2kernel/Regularizer_5/Square/ReadVariableOp:value:0*
T0**
_output_shapes
:@@w
kernel/Regularizer_5/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
kernel/Regularizer_5/SumSumkernel/Regularizer_5/Square:y:0#kernel/Regularizer_5/Const:output:0*
T0*
_output_shapes
: _
kernel/Regularizer_5/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;�
kernel/Regularizer_5/mulMul#kernel/Regularizer_5/mul/x:output:0!kernel/Regularizer_5/Sum:output:0*
T0*
_output_shapes
: �
*kernel/Regularizer_6/Square/ReadVariableOpReadVariableOpconv3d_6_10621*+
_output_shapes
:@�*
dtype0�
kernel/Regularizer_6/SquareSquare2kernel/Regularizer_6/Square/ReadVariableOp:value:0*
T0*+
_output_shapes
:@�w
kernel/Regularizer_6/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
kernel/Regularizer_6/SumSumkernel/Regularizer_6/Square:y:0#kernel/Regularizer_6/Const:output:0*
T0*
_output_shapes
: _
kernel/Regularizer_6/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;�
kernel/Regularizer_6/mulMul#kernel/Regularizer_6/mul/x:output:0!kernel/Regularizer_6/Sum:output:0*
T0*
_output_shapes
: �
*kernel/Regularizer_7/Square/ReadVariableOpReadVariableOpconv3d_7_10644*,
_output_shapes
:��*
dtype0�
kernel/Regularizer_7/SquareSquare2kernel/Regularizer_7/Square/ReadVariableOp:value:0*
T0*,
_output_shapes
:��w
kernel/Regularizer_7/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
kernel/Regularizer_7/SumSumkernel/Regularizer_7/Square:y:0#kernel/Regularizer_7/Const:output:0*
T0*
_output_shapes
: _
kernel/Regularizer_7/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;�
kernel/Regularizer_7/mulMul#kernel/Regularizer_7/mul/x:output:0!kernel/Regularizer_7/Sum:output:0*
T0*
_output_shapes
: u
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^conv3d/StatefulPartitionedCall!^conv3d_1/StatefulPartitionedCall!^conv3d_2/StatefulPartitionedCall!^conv3d_3/StatefulPartitionedCall!^conv3d_4/StatefulPartitionedCall!^conv3d_5/StatefulPartitionedCall!^conv3d_6/StatefulPartitionedCall!^conv3d_7/StatefulPartitionedCall^dense/StatefulPartitionedCall)^kernel/Regularizer/Square/ReadVariableOp+^kernel/Regularizer_1/Square/ReadVariableOp+^kernel/Regularizer_2/Square/ReadVariableOp+^kernel/Regularizer_3/Square/ReadVariableOp+^kernel/Regularizer_4/Square/ReadVariableOp+^kernel/Regularizer_5/Square/ReadVariableOp+^kernel/Regularizer_6/Square/ReadVariableOp+^kernel/Regularizer_7/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:���������22: : : : : : : : : : : : : : : : : : 2@
conv3d/StatefulPartitionedCallconv3d/StatefulPartitionedCall2D
 conv3d_1/StatefulPartitionedCall conv3d_1/StatefulPartitionedCall2D
 conv3d_2/StatefulPartitionedCall conv3d_2/StatefulPartitionedCall2D
 conv3d_3/StatefulPartitionedCall conv3d_3/StatefulPartitionedCall2D
 conv3d_4/StatefulPartitionedCall conv3d_4/StatefulPartitionedCall2D
 conv3d_5/StatefulPartitionedCall conv3d_5/StatefulPartitionedCall2D
 conv3d_6/StatefulPartitionedCall conv3d_6/StatefulPartitionedCall2D
 conv3d_7/StatefulPartitionedCall conv3d_7/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2T
(kernel/Regularizer/Square/ReadVariableOp(kernel/Regularizer/Square/ReadVariableOp2X
*kernel/Regularizer_1/Square/ReadVariableOp*kernel/Regularizer_1/Square/ReadVariableOp2X
*kernel/Regularizer_2/Square/ReadVariableOp*kernel/Regularizer_2/Square/ReadVariableOp2X
*kernel/Regularizer_3/Square/ReadVariableOp*kernel/Regularizer_3/Square/ReadVariableOp2X
*kernel/Regularizer_4/Square/ReadVariableOp*kernel/Regularizer_4/Square/ReadVariableOp2X
*kernel/Regularizer_5/Square/ReadVariableOp*kernel/Regularizer_5/Square/ReadVariableOp2X
*kernel/Regularizer_6/Square/ReadVariableOp*kernel/Regularizer_6/Square/ReadVariableOp2X
*kernel/Regularizer_7/Square/ReadVariableOp*kernel/Regularizer_7/Square/ReadVariableOp:[ W
3
_output_shapes!
:���������22
 
_user_specified_nameinputs
�
�
(__inference_conv3d_6_layer_call_fn_11967

inputs&
unknown:@�
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :����������*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *L
fGRE
C__inference_conv3d_6_layer_call_and_return_conditional_losses_10620|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������@: : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:���������@
 
_user_specified_nameinputs
�

�
__inference_loss_fn_7_12138Q
1kernel_regularizer_square_readvariableop_resource:��
identity��(kernel/Regularizer/Square/ReadVariableOp�
(kernel/Regularizer/Square/ReadVariableOpReadVariableOp1kernel_regularizer_square_readvariableop_resource*,
_output_shapes
:��*
dtype0�
kernel/Regularizer/SquareSquare0kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*,
_output_shapes
:��u
kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
kernel/Regularizer/SumSumkernel/Regularizer/Square:y:0!kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: ]
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;�
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: X
IdentityIdentitykernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: q
NoOpNoOp)^kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2T
(kernel/Regularizer/Square/ReadVariableOp(kernel/Regularizer/Square/ReadVariableOp
�
�
C__inference_conv3d_5_layer_call_and_return_conditional_losses_11948

inputs<
conv3d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�Conv3D/ReadVariableOp�(kernel/Regularizer/Square/ReadVariableOp�
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:@@*
dtype0�
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������@*
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������@\
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:���������@�
(kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:@@*
dtype0�
kernel/Regularizer/SquareSquare0kernel/Regularizer/Square/ReadVariableOp:value:0*
T0**
_output_shapes
:@@u
kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
kernel/Regularizer/SumSumkernel/Regularizer/Square:y:0!kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: ]
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;�
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:���������@�
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp)^kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp2T
(kernel/Regularizer/Square/ReadVariableOp(kernel/Regularizer/Square/ReadVariableOp:[ W
3
_output_shapes!
:���������@
 
_user_specified_nameinputs
�

�
__inference_loss_fn_3_12094O
1kernel_regularizer_square_readvariableop_resource:  
identity��(kernel/Regularizer/Square/ReadVariableOp�
(kernel/Regularizer/Square/ReadVariableOpReadVariableOp1kernel_regularizer_square_readvariableop_resource**
_output_shapes
:  *
dtype0�
kernel/Regularizer/SquareSquare0kernel/Regularizer/Square/ReadVariableOp:value:0*
T0**
_output_shapes
:  u
kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
kernel/Regularizer/SumSumkernel/Regularizer/Square:y:0!kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: ]
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;�
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: X
IdentityIdentitykernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: q
NoOpNoOp)^kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2T
(kernel/Regularizer/Square/ReadVariableOp(kernel/Regularizer/Square/ReadVariableOp
�
�
%__inference_model_layer_call_fn_11532

inputs%
unknown:
	unknown_0:'
	unknown_1:
	unknown_2:'
	unknown_3: 
	unknown_4: '
	unknown_5:  
	unknown_6: '
	unknown_7: @
	unknown_8:@'
	unknown_9:@@

unknown_10:@)

unknown_11:@�

unknown_12:	�*

unknown_13:��

unknown_14:	�

unknown_15:	�	

unknown_16:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*4
_read_only_resource_inputs
	
*4
config_proto$"

CPU

GPU(2*0J 8� *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_11003o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:���������22: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:���������22
 
_user_specified_nameinputs
�
�
C__inference_conv3d_1_layer_call_and_return_conditional_losses_10502

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv3D/ReadVariableOp�(kernel/Regularizer/Square/ReadVariableOp�
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������22*
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������22\
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:���������22�
(kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
kernel/Regularizer/SquareSquare0kernel/Regularizer/Square/ReadVariableOp:value:0*
T0**
_output_shapes
:u
kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
kernel/Regularizer/SumSumkernel/Regularizer/Square:y:0!kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: ]
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;�
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:���������22�
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp)^kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������22: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp2T
(kernel/Regularizer/Square/ReadVariableOp(kernel/Regularizer/Square/ReadVariableOp:[ W
3
_output_shapes!
:���������22
 
_user_specified_nameinputs
�
�
#__inference_signature_wrapper_11402
input_1%
unknown:
	unknown_0:'
	unknown_1:
	unknown_2:'
	unknown_3: 
	unknown_4: '
	unknown_5:  
	unknown_6: '
	unknown_7: @
	unknown_8:@'
	unknown_9:@@

unknown_10:@)

unknown_11:@�

unknown_12:	�*

unknown_13:��

unknown_14:	�

unknown_15:	�	

unknown_16:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*4
_read_only_resource_inputs
	
*4
config_proto$"

CPU

GPU(2*0J 8� *)
f$R"
 __inference__wrapped_model_10407o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:���������22: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
3
_output_shapes!
:���������22
!
_user_specified_name	input_1
�d
�
 __inference__wrapped_model_10407
input_1I
+model_conv3d_conv3d_readvariableop_resource::
,model_conv3d_biasadd_readvariableop_resource:K
-model_conv3d_1_conv3d_readvariableop_resource:<
.model_conv3d_1_biasadd_readvariableop_resource:K
-model_conv3d_2_conv3d_readvariableop_resource: <
.model_conv3d_2_biasadd_readvariableop_resource: K
-model_conv3d_3_conv3d_readvariableop_resource:  <
.model_conv3d_3_biasadd_readvariableop_resource: K
-model_conv3d_4_conv3d_readvariableop_resource: @<
.model_conv3d_4_biasadd_readvariableop_resource:@K
-model_conv3d_5_conv3d_readvariableop_resource:@@<
.model_conv3d_5_biasadd_readvariableop_resource:@L
-model_conv3d_6_conv3d_readvariableop_resource:@�=
.model_conv3d_6_biasadd_readvariableop_resource:	�M
-model_conv3d_7_conv3d_readvariableop_resource:��=
.model_conv3d_7_biasadd_readvariableop_resource:	�=
*model_dense_matmul_readvariableop_resource:	�	9
+model_dense_biasadd_readvariableop_resource:
identity��#model/conv3d/BiasAdd/ReadVariableOp�"model/conv3d/Conv3D/ReadVariableOp�%model/conv3d_1/BiasAdd/ReadVariableOp�$model/conv3d_1/Conv3D/ReadVariableOp�%model/conv3d_2/BiasAdd/ReadVariableOp�$model/conv3d_2/Conv3D/ReadVariableOp�%model/conv3d_3/BiasAdd/ReadVariableOp�$model/conv3d_3/Conv3D/ReadVariableOp�%model/conv3d_4/BiasAdd/ReadVariableOp�$model/conv3d_4/Conv3D/ReadVariableOp�%model/conv3d_5/BiasAdd/ReadVariableOp�$model/conv3d_5/Conv3D/ReadVariableOp�%model/conv3d_6/BiasAdd/ReadVariableOp�$model/conv3d_6/Conv3D/ReadVariableOp�%model/conv3d_7/BiasAdd/ReadVariableOp�$model/conv3d_7/Conv3D/ReadVariableOp�"model/dense/BiasAdd/ReadVariableOp�!model/dense/MatMul/ReadVariableOp�
"model/conv3d/Conv3D/ReadVariableOpReadVariableOp+model_conv3d_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
model/conv3d/Conv3DConv3Dinput_1*model/conv3d/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������22*
paddingSAME*
strides	
�
#model/conv3d/BiasAdd/ReadVariableOpReadVariableOp,model_conv3d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model/conv3d/BiasAddBiasAddmodel/conv3d/Conv3D:output:0+model/conv3d/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������22v
model/conv3d/ReluRelumodel/conv3d/BiasAdd:output:0*
T0*3
_output_shapes!
:���������22�
$model/conv3d_1/Conv3D/ReadVariableOpReadVariableOp-model_conv3d_1_conv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
model/conv3d_1/Conv3DConv3Dmodel/conv3d/Relu:activations:0,model/conv3d_1/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������22*
paddingSAME*
strides	
�
%model/conv3d_1/BiasAdd/ReadVariableOpReadVariableOp.model_conv3d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model/conv3d_1/BiasAddBiasAddmodel/conv3d_1/Conv3D:output:0-model/conv3d_1/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������22z
model/conv3d_1/ReluRelumodel/conv3d_1/BiasAdd:output:0*
T0*3
_output_shapes!
:���������22�
model/max_pooling3d/MaxPool3D	MaxPool3D!model/conv3d_1/Relu:activations:0*
T0*3
_output_shapes!
:���������
*
ksize	
*
paddingVALID*
strides	
�
$model/conv3d_2/Conv3D/ReadVariableOpReadVariableOp-model_conv3d_2_conv3d_readvariableop_resource**
_output_shapes
: *
dtype0�
model/conv3d_2/Conv3DConv3D&model/max_pooling3d/MaxPool3D:output:0,model/conv3d_2/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������
 *
paddingSAME*
strides	
�
%model/conv3d_2/BiasAdd/ReadVariableOpReadVariableOp.model_conv3d_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
model/conv3d_2/BiasAddBiasAddmodel/conv3d_2/Conv3D:output:0-model/conv3d_2/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������
 z
model/conv3d_2/ReluRelumodel/conv3d_2/BiasAdd:output:0*
T0*3
_output_shapes!
:���������
 �
$model/conv3d_3/Conv3D/ReadVariableOpReadVariableOp-model_conv3d_3_conv3d_readvariableop_resource**
_output_shapes
:  *
dtype0�
model/conv3d_3/Conv3DConv3D!model/conv3d_2/Relu:activations:0,model/conv3d_3/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������
 *
paddingSAME*
strides	
�
%model/conv3d_3/BiasAdd/ReadVariableOpReadVariableOp.model_conv3d_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
model/conv3d_3/BiasAddBiasAddmodel/conv3d_3/Conv3D:output:0-model/conv3d_3/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������
 z
model/conv3d_3/ReluRelumodel/conv3d_3/BiasAdd:output:0*
T0*3
_output_shapes!
:���������
 �
model/max_pooling3d_1/MaxPool3D	MaxPool3D!model/conv3d_3/Relu:activations:0*
T0*3
_output_shapes!
:��������� *
ksize	
*
paddingVALID*
strides	
�
$model/conv3d_4/Conv3D/ReadVariableOpReadVariableOp-model_conv3d_4_conv3d_readvariableop_resource**
_output_shapes
: @*
dtype0�
model/conv3d_4/Conv3DConv3D(model/max_pooling3d_1/MaxPool3D:output:0,model/conv3d_4/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������@*
paddingSAME*
strides	
�
%model/conv3d_4/BiasAdd/ReadVariableOpReadVariableOp.model_conv3d_4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
model/conv3d_4/BiasAddBiasAddmodel/conv3d_4/Conv3D:output:0-model/conv3d_4/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������@z
model/conv3d_4/ReluRelumodel/conv3d_4/BiasAdd:output:0*
T0*3
_output_shapes!
:���������@�
$model/conv3d_5/Conv3D/ReadVariableOpReadVariableOp-model_conv3d_5_conv3d_readvariableop_resource**
_output_shapes
:@@*
dtype0�
model/conv3d_5/Conv3DConv3D!model/conv3d_4/Relu:activations:0,model/conv3d_5/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������@*
paddingSAME*
strides	
�
%model/conv3d_5/BiasAdd/ReadVariableOpReadVariableOp.model_conv3d_5_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
model/conv3d_5/BiasAddBiasAddmodel/conv3d_5/Conv3D:output:0-model/conv3d_5/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������@z
model/conv3d_5/ReluRelumodel/conv3d_5/BiasAdd:output:0*
T0*3
_output_shapes!
:���������@�
model/max_pooling3d_2/MaxPool3D	MaxPool3D!model/conv3d_5/Relu:activations:0*
T0*3
_output_shapes!
:���������@*
ksize	
*
paddingVALID*
strides	
�
$model/conv3d_6/Conv3D/ReadVariableOpReadVariableOp-model_conv3d_6_conv3d_readvariableop_resource*+
_output_shapes
:@�*
dtype0�
model/conv3d_6/Conv3DConv3D(model/max_pooling3d_2/MaxPool3D:output:0,model/conv3d_6/Conv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :����������*
paddingSAME*
strides	
�
%model/conv3d_6/BiasAdd/ReadVariableOpReadVariableOp.model_conv3d_6_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
model/conv3d_6/BiasAddBiasAddmodel/conv3d_6/Conv3D:output:0-model/conv3d_6/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :����������{
model/conv3d_6/ReluRelumodel/conv3d_6/BiasAdd:output:0*
T0*4
_output_shapes"
 :�����������
$model/conv3d_7/Conv3D/ReadVariableOpReadVariableOp-model_conv3d_7_conv3d_readvariableop_resource*,
_output_shapes
:��*
dtype0�
model/conv3d_7/Conv3DConv3D!model/conv3d_6/Relu:activations:0,model/conv3d_7/Conv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :����������*
paddingSAME*
strides	
�
%model/conv3d_7/BiasAdd/ReadVariableOpReadVariableOp.model_conv3d_7_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
model/conv3d_7/BiasAddBiasAddmodel/conv3d_7/Conv3D:output:0-model/conv3d_7/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :����������{
model/conv3d_7/ReluRelumodel/conv3d_7/BiasAdd:output:0*
T0*4
_output_shapes"
 :�����������
model/max_pooling3d_3/MaxPool3D	MaxPool3D!model/conv3d_7/Relu:activations:0*
T0*4
_output_shapes"
 :����������*
ksize	
*
paddingVALID*
strides	
d
model/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����  �
model/flatten/ReshapeReshape(model/max_pooling3d_3/MaxPool3D:output:0model/flatten/Const:output:0*
T0*(
_output_shapes
:����������	�
!model/dense/MatMul/ReadVariableOpReadVariableOp*model_dense_matmul_readvariableop_resource*
_output_shapes
:	�	*
dtype0�
model/dense/MatMulMatMulmodel/flatten/Reshape:output:0)model/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
"model/dense/BiasAdd/ReadVariableOpReadVariableOp+model_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model/dense/BiasAddBiasAddmodel/dense/MatMul:product:0*model/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������k
IdentityIdentitymodel/dense/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp$^model/conv3d/BiasAdd/ReadVariableOp#^model/conv3d/Conv3D/ReadVariableOp&^model/conv3d_1/BiasAdd/ReadVariableOp%^model/conv3d_1/Conv3D/ReadVariableOp&^model/conv3d_2/BiasAdd/ReadVariableOp%^model/conv3d_2/Conv3D/ReadVariableOp&^model/conv3d_3/BiasAdd/ReadVariableOp%^model/conv3d_3/Conv3D/ReadVariableOp&^model/conv3d_4/BiasAdd/ReadVariableOp%^model/conv3d_4/Conv3D/ReadVariableOp&^model/conv3d_5/BiasAdd/ReadVariableOp%^model/conv3d_5/Conv3D/ReadVariableOp&^model/conv3d_6/BiasAdd/ReadVariableOp%^model/conv3d_6/Conv3D/ReadVariableOp&^model/conv3d_7/BiasAdd/ReadVariableOp%^model/conv3d_7/Conv3D/ReadVariableOp#^model/dense/BiasAdd/ReadVariableOp"^model/dense/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:���������22: : : : : : : : : : : : : : : : : : 2J
#model/conv3d/BiasAdd/ReadVariableOp#model/conv3d/BiasAdd/ReadVariableOp2H
"model/conv3d/Conv3D/ReadVariableOp"model/conv3d/Conv3D/ReadVariableOp2N
%model/conv3d_1/BiasAdd/ReadVariableOp%model/conv3d_1/BiasAdd/ReadVariableOp2L
$model/conv3d_1/Conv3D/ReadVariableOp$model/conv3d_1/Conv3D/ReadVariableOp2N
%model/conv3d_2/BiasAdd/ReadVariableOp%model/conv3d_2/BiasAdd/ReadVariableOp2L
$model/conv3d_2/Conv3D/ReadVariableOp$model/conv3d_2/Conv3D/ReadVariableOp2N
%model/conv3d_3/BiasAdd/ReadVariableOp%model/conv3d_3/BiasAdd/ReadVariableOp2L
$model/conv3d_3/Conv3D/ReadVariableOp$model/conv3d_3/Conv3D/ReadVariableOp2N
%model/conv3d_4/BiasAdd/ReadVariableOp%model/conv3d_4/BiasAdd/ReadVariableOp2L
$model/conv3d_4/Conv3D/ReadVariableOp$model/conv3d_4/Conv3D/ReadVariableOp2N
%model/conv3d_5/BiasAdd/ReadVariableOp%model/conv3d_5/BiasAdd/ReadVariableOp2L
$model/conv3d_5/Conv3D/ReadVariableOp$model/conv3d_5/Conv3D/ReadVariableOp2N
%model/conv3d_6/BiasAdd/ReadVariableOp%model/conv3d_6/BiasAdd/ReadVariableOp2L
$model/conv3d_6/Conv3D/ReadVariableOp$model/conv3d_6/Conv3D/ReadVariableOp2N
%model/conv3d_7/BiasAdd/ReadVariableOp%model/conv3d_7/BiasAdd/ReadVariableOp2L
$model/conv3d_7/Conv3D/ReadVariableOp$model/conv3d_7/Conv3D/ReadVariableOp2H
"model/dense/BiasAdd/ReadVariableOp"model/dense/BiasAdd/ReadVariableOp2F
!model/dense/MatMul/ReadVariableOp!model/dense/MatMul/ReadVariableOp:\ X
3
_output_shapes!
:���������22
!
_user_specified_name	input_1
�
�
%__inference_model_layer_call_fn_11491

inputs%
unknown:
	unknown_0:'
	unknown_1:
	unknown_2:'
	unknown_3: 
	unknown_4: '
	unknown_5:  
	unknown_6: '
	unknown_7: @
	unknown_8:@'
	unknown_9:@@

unknown_10:@)

unknown_11:@�

unknown_12:	�*

unknown_13:��

unknown_14:	�

unknown_15:	�	

unknown_16:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*4
_read_only_resource_inputs
	
*4
config_proto$"

CPU

GPU(2*0J 8� *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_10723o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:���������22: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:���������22
 
_user_specified_nameinputs
�
f
J__inference_max_pooling3d_1_layer_call_and_return_conditional_losses_10428

inputs
identity�
	MaxPool3D	MaxPool3Dinputs*
T0*W
_output_shapesE
C:A���������������������������������������������*
ksize	
*
paddingVALID*
strides	
�
IdentityIdentityMaxPool3D:output:0*
T0*W
_output_shapesE
C:A���������������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A���������������������������������������������: {
W
_output_shapesE
C:A���������������������������������������������
 
_user_specified_nameinputs
�	
�
@__inference_dense_layer_call_and_return_conditional_losses_10668

inputs1
matmul_readvariableop_resource:	�	-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�	*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������	
 
_user_specified_nameinputs
�
f
J__inference_max_pooling3d_2_layer_call_and_return_conditional_losses_11958

inputs
identity�
	MaxPool3D	MaxPool3Dinputs*
T0*W
_output_shapesE
C:A���������������������������������������������*
ksize	
*
paddingVALID*
strides	
�
IdentityIdentityMaxPool3D:output:0*
T0*W
_output_shapesE
C:A���������������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A���������������������������������������������: {
W
_output_shapesE
C:A���������������������������������������������
 
_user_specified_nameinputs
�
�
A__inference_conv3d_layer_call_and_return_conditional_losses_11798

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv3D/ReadVariableOp�(kernel/Regularizer/Square/ReadVariableOp�
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������22*
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������22\
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:���������22�
(kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
kernel/Regularizer/SquareSquare0kernel/Regularizer/Square/ReadVariableOp:value:0*
T0**
_output_shapes
:u
kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
kernel/Regularizer/SumSumkernel/Regularizer/Square:y:0!kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: ]
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;�
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:���������22�
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp)^kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������22: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp2T
(kernel/Regularizer/Square/ReadVariableOp(kernel/Regularizer/Square/ReadVariableOp:[ W
3
_output_shapes!
:���������22
 
_user_specified_nameinputs
�

�
__inference_loss_fn_6_12127P
1kernel_regularizer_square_readvariableop_resource:@�
identity��(kernel/Regularizer/Square/ReadVariableOp�
(kernel/Regularizer/Square/ReadVariableOpReadVariableOp1kernel_regularizer_square_readvariableop_resource*+
_output_shapes
:@�*
dtype0�
kernel/Regularizer/SquareSquare0kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*+
_output_shapes
:@�u
kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
kernel/Regularizer/SumSumkernel/Regularizer/Square:y:0!kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: ]
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;�
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: X
IdentityIdentitykernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: q
NoOpNoOp)^kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2T
(kernel/Regularizer/Square/ReadVariableOp(kernel/Regularizer/Square/ReadVariableOp
�
f
J__inference_max_pooling3d_1_layer_call_and_return_conditional_losses_11896

inputs
identity�
	MaxPool3D	MaxPool3Dinputs*
T0*W
_output_shapesE
C:A���������������������������������������������*
ksize	
*
paddingVALID*
strides	
�
IdentityIdentityMaxPool3D:output:0*
T0*W
_output_shapesE
C:A���������������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A���������������������������������������������: {
W
_output_shapesE
C:A���������������������������������������������
 
_user_specified_nameinputs
�
�
C__inference_conv3d_4_layer_call_and_return_conditional_losses_10573

inputs<
conv3d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�Conv3D/ReadVariableOp�(kernel/Regularizer/Square/ReadVariableOp�
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
: @*
dtype0�
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������@*
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������@\
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:���������@�
(kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
: @*
dtype0�
kernel/Regularizer/SquareSquare0kernel/Regularizer/Square/ReadVariableOp:value:0*
T0**
_output_shapes
: @u
kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
kernel/Regularizer/SumSumkernel/Regularizer/Square:y:0!kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: ]
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;�
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:���������@�
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp)^kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp2T
(kernel/Regularizer/Square/ReadVariableOp(kernel/Regularizer/Square/ReadVariableOp:[ W
3
_output_shapes!
:��������� 
 
_user_specified_nameinputs
�
K
/__inference_max_pooling3d_3_layer_call_fn_12015

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *W
_output_shapesE
C:A���������������������������������������������* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8� *S
fNRL
J__inference_max_pooling3d_3_layer_call_and_return_conditional_losses_10452�
IdentityIdentityPartitionedCall:output:0*
T0*W
_output_shapesE
C:A���������������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A���������������������������������������������: {
W
_output_shapesE
C:A���������������������������������������������
 
_user_specified_nameinputs
�
�
(__inference_conv3d_1_layer_call_fn_11807

inputs%
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������22*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *L
fGRE
C__inference_conv3d_1_layer_call_and_return_conditional_losses_10502{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:���������22`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������22: : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:���������22
 
_user_specified_nameinputs
�
�
C__inference_conv3d_7_layer_call_and_return_conditional_losses_12010

inputs>
conv3d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv3D/ReadVariableOp�(kernel/Regularizer/Square/ReadVariableOp�
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource*,
_output_shapes
:��*
dtype0�
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :����������*
paddingSAME*
strides	
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :����������]
ReluReluBiasAdd:output:0*
T0*4
_output_shapes"
 :�����������
(kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv3d_readvariableop_resource*,
_output_shapes
:��*
dtype0�
kernel/Regularizer/SquareSquare0kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*,
_output_shapes
:��u
kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
kernel/Regularizer/SumSumkernel/Regularizer/Square:y:0!kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: ]
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;�
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: n
IdentityIdentityRelu:activations:0^NoOp*
T0*4
_output_shapes"
 :�����������
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp)^kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp2T
(kernel/Regularizer/Square/ReadVariableOp(kernel/Regularizer/Square/ReadVariableOp:\ X
4
_output_shapes"
 :����������
 
_user_specified_nameinputs
�
�
C__inference_conv3d_6_layer_call_and_return_conditional_losses_10620

inputs=
conv3d_readvariableop_resource:@�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv3D/ReadVariableOp�(kernel/Regularizer/Square/ReadVariableOp�
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource*+
_output_shapes
:@�*
dtype0�
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :����������*
paddingSAME*
strides	
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :����������]
ReluReluBiasAdd:output:0*
T0*4
_output_shapes"
 :�����������
(kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv3d_readvariableop_resource*+
_output_shapes
:@�*
dtype0�
kernel/Regularizer/SquareSquare0kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*+
_output_shapes
:@�u
kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
kernel/Regularizer/SumSumkernel/Regularizer/Square:y:0!kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: ]
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;�
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: n
IdentityIdentityRelu:activations:0^NoOp*
T0*4
_output_shapes"
 :�����������
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp)^kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp2T
(kernel/Regularizer/Square/ReadVariableOp(kernel/Regularizer/Square/ReadVariableOp:[ W
3
_output_shapes!
:���������@
 
_user_specified_nameinputs
�
�
C__inference_conv3d_2_layer_call_and_return_conditional_losses_10526

inputs<
conv3d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�Conv3D/ReadVariableOp�(kernel/Regularizer/Square/ReadVariableOp�
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
: *
dtype0�
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������
 *
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������
 \
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:���������
 �
(kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
: *
dtype0�
kernel/Regularizer/SquareSquare0kernel/Regularizer/Square/ReadVariableOp:value:0*
T0**
_output_shapes
: u
kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
kernel/Regularizer/SumSumkernel/Regularizer/Square:y:0!kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: ]
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;�
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:���������
 �
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp)^kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp2T
(kernel/Regularizer/Square/ReadVariableOp(kernel/Regularizer/Square/ReadVariableOp:[ W
3
_output_shapes!
:���������

 
_user_specified_nameinputs
�
K
/__inference_max_pooling3d_1_layer_call_fn_11891

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *W
_output_shapesE
C:A���������������������������������������������* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8� *S
fNRL
J__inference_max_pooling3d_1_layer_call_and_return_conditional_losses_10428�
IdentityIdentityPartitionedCall:output:0*
T0*W
_output_shapesE
C:A���������������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A���������������������������������������������: {
W
_output_shapesE
C:A���������������������������������������������
 
_user_specified_nameinputs
�
d
H__inference_max_pooling3d_layer_call_and_return_conditional_losses_10416

inputs
identity�
	MaxPool3D	MaxPool3Dinputs*
T0*W
_output_shapesE
C:A���������������������������������������������*
ksize	
*
paddingVALID*
strides	
�
IdentityIdentityMaxPool3D:output:0*
T0*W
_output_shapesE
C:A���������������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A���������������������������������������������: {
W
_output_shapesE
C:A���������������������������������������������
 
_user_specified_nameinputs
�@
�
__inference__traced_save_12254
file_prefix,
(savev2_conv3d_kernel_read_readvariableop*
&savev2_conv3d_bias_read_readvariableop.
*savev2_conv3d_1_kernel_read_readvariableop,
(savev2_conv3d_1_bias_read_readvariableop.
*savev2_conv3d_2_kernel_read_readvariableop,
(savev2_conv3d_2_bias_read_readvariableop.
*savev2_conv3d_3_kernel_read_readvariableop,
(savev2_conv3d_3_bias_read_readvariableop.
*savev2_conv3d_4_kernel_read_readvariableop,
(savev2_conv3d_4_bias_read_readvariableop.
*savev2_conv3d_5_kernel_read_readvariableop,
(savev2_conv3d_5_bias_read_readvariableop.
*savev2_conv3d_6_kernel_read_readvariableop,
(savev2_conv3d_6_bias_read_readvariableop.
*savev2_conv3d_7_kernel_read_readvariableop,
(savev2_conv3d_7_bias_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop2
.savev2_adam_dense_kernel_m_read_readvariableop0
,savev2_adam_dense_bias_m_read_readvariableop2
.savev2_adam_dense_kernel_v_read_readvariableop0
,savev2_adam_dense_bias_v_read_readvariableop
savev2_const

identity_1��MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
: *
dtype0*�
value�B� B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
: *
dtype0*S
valueJBH B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0(savev2_conv3d_kernel_read_readvariableop&savev2_conv3d_bias_read_readvariableop*savev2_conv3d_1_kernel_read_readvariableop(savev2_conv3d_1_bias_read_readvariableop*savev2_conv3d_2_kernel_read_readvariableop(savev2_conv3d_2_bias_read_readvariableop*savev2_conv3d_3_kernel_read_readvariableop(savev2_conv3d_3_bias_read_readvariableop*savev2_conv3d_4_kernel_read_readvariableop(savev2_conv3d_4_bias_read_readvariableop*savev2_conv3d_5_kernel_read_readvariableop(savev2_conv3d_5_bias_read_readvariableop*savev2_conv3d_6_kernel_read_readvariableop(savev2_conv3d_6_bias_read_readvariableop*savev2_conv3d_7_kernel_read_readvariableop(savev2_conv3d_7_bias_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop.savev2_adam_dense_kernel_m_read_readvariableop,savev2_adam_dense_bias_m_read_readvariableop.savev2_adam_dense_kernel_v_read_readvariableop,savev2_adam_dense_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *.
dtypes$
"2 	�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*�
_input_shapes�
�: ::::: : :  : : @:@:@@:@:@�:�:��:�:	�	:: : : : : : : : : :	�	::	�	:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:0,
*
_output_shapes
:: 

_output_shapes
::0,
*
_output_shapes
:: 

_output_shapes
::0,
*
_output_shapes
: : 

_output_shapes
: :0,
*
_output_shapes
:  : 

_output_shapes
: :0	,
*
_output_shapes
: @: 


_output_shapes
:@:0,
*
_output_shapes
:@@: 

_output_shapes
:@:1-
+
_output_shapes
:@�:!

_output_shapes	
:�:2.
,
_output_shapes
:��:!

_output_shapes	
:�:%!

_output_shapes
:	�	: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	�	: 

_output_shapes
::%!

_output_shapes
:	�	: 

_output_shapes
:: 

_output_shapes
: 
�y
�
!__inference__traced_restore_12357
file_prefix<
assignvariableop_conv3d_kernel:,
assignvariableop_1_conv3d_bias:@
"assignvariableop_2_conv3d_1_kernel:.
 assignvariableop_3_conv3d_1_bias:@
"assignvariableop_4_conv3d_2_kernel: .
 assignvariableop_5_conv3d_2_bias: @
"assignvariableop_6_conv3d_3_kernel:  .
 assignvariableop_7_conv3d_3_bias: @
"assignvariableop_8_conv3d_4_kernel: @.
 assignvariableop_9_conv3d_4_bias:@A
#assignvariableop_10_conv3d_5_kernel:@@/
!assignvariableop_11_conv3d_5_bias:@B
#assignvariableop_12_conv3d_6_kernel:@�0
!assignvariableop_13_conv3d_6_bias:	�C
#assignvariableop_14_conv3d_7_kernel:��0
!assignvariableop_15_conv3d_7_bias:	�3
 assignvariableop_16_dense_kernel:	�	,
assignvariableop_17_dense_bias:'
assignvariableop_18_adam_iter:	 )
assignvariableop_19_adam_beta_1: )
assignvariableop_20_adam_beta_2: (
assignvariableop_21_adam_decay: 0
&assignvariableop_22_adam_learning_rate: %
assignvariableop_23_total_1: %
assignvariableop_24_count_1: #
assignvariableop_25_total: #
assignvariableop_26_count: :
'assignvariableop_27_adam_dense_kernel_m:	�	3
%assignvariableop_28_adam_dense_bias_m::
'assignvariableop_29_adam_dense_kernel_v:	�	3
%assignvariableop_30_adam_dense_bias_v:
identity_32��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
: *
dtype0*�
value�B� B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
: *
dtype0*S
valueJBH B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�::::::::::::::::::::::::::::::::*.
dtypes$
"2 	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOpassignvariableop_conv3d_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOpassignvariableop_1_conv3d_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp"assignvariableop_2_conv3d_1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp assignvariableop_3_conv3d_1_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp"assignvariableop_4_conv3d_2_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp assignvariableop_5_conv3d_2_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp"assignvariableop_6_conv3d_3_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp assignvariableop_7_conv3d_3_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp"assignvariableop_8_conv3d_4_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp assignvariableop_9_conv3d_4_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOp#assignvariableop_10_conv3d_5_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOp!assignvariableop_11_conv3d_5_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOp#assignvariableop_12_conv3d_6_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOp!assignvariableop_13_conv3d_6_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOp#assignvariableop_14_conv3d_7_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOp!assignvariableop_15_conv3d_7_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOp assignvariableop_16_dense_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOpassignvariableop_17_dense_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_18AssignVariableOpassignvariableop_18_adam_iterIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOpassignvariableop_19_adam_beta_1Identity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOpassignvariableop_20_adam_beta_2Identity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOpassignvariableop_21_adam_decayIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOp&assignvariableop_22_adam_learning_rateIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOpassignvariableop_23_total_1Identity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_24AssignVariableOpassignvariableop_24_count_1Identity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_25AssignVariableOpassignvariableop_25_totalIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_26AssignVariableOpassignvariableop_26_countIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_27AssignVariableOp'assignvariableop_27_adam_dense_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_28AssignVariableOp%assignvariableop_28_adam_dense_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_29AssignVariableOp'assignvariableop_29_adam_dense_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_30AssignVariableOp%assignvariableop_30_adam_dense_bias_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 �
Identity_31Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_32IdentityIdentity_31:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_32Identity_32:output:0*S
_input_shapesB
@: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�

�
__inference_loss_fn_5_12116O
1kernel_regularizer_square_readvariableop_resource:@@
identity��(kernel/Regularizer/Square/ReadVariableOp�
(kernel/Regularizer/Square/ReadVariableOpReadVariableOp1kernel_regularizer_square_readvariableop_resource**
_output_shapes
:@@*
dtype0�
kernel/Regularizer/SquareSquare0kernel/Regularizer/Square/ReadVariableOp:value:0*
T0**
_output_shapes
:@@u
kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
kernel/Regularizer/SumSumkernel/Regularizer/Square:y:0!kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: ]
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;�
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: X
IdentityIdentitykernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: q
NoOpNoOp)^kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2T
(kernel/Regularizer/Square/ReadVariableOp(kernel/Regularizer/Square/ReadVariableOp
�
�
&__inference_conv3d_layer_call_fn_11781

inputs%
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������22*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *J
fERC
A__inference_conv3d_layer_call_and_return_conditional_losses_10479{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:���������22`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������22: : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:���������22
 
_user_specified_nameinputs
�z
�
@__inference_model_layer_call_and_return_conditional_losses_11185
input_1*
conv3d_11086:
conv3d_11088:,
conv3d_1_11091:
conv3d_1_11093:,
conv3d_2_11097: 
conv3d_2_11099: ,
conv3d_3_11102:  
conv3d_3_11104: ,
conv3d_4_11108: @
conv3d_4_11110:@,
conv3d_5_11113:@@
conv3d_5_11115:@-
conv3d_6_11119:@�
conv3d_6_11121:	�.
conv3d_7_11124:��
conv3d_7_11126:	�
dense_11131:	�	
dense_11133:
identity��conv3d/StatefulPartitionedCall� conv3d_1/StatefulPartitionedCall� conv3d_2/StatefulPartitionedCall� conv3d_3/StatefulPartitionedCall� conv3d_4/StatefulPartitionedCall� conv3d_5/StatefulPartitionedCall� conv3d_6/StatefulPartitionedCall� conv3d_7/StatefulPartitionedCall�dense/StatefulPartitionedCall�(kernel/Regularizer/Square/ReadVariableOp�*kernel/Regularizer_1/Square/ReadVariableOp�*kernel/Regularizer_2/Square/ReadVariableOp�*kernel/Regularizer_3/Square/ReadVariableOp�*kernel/Regularizer_4/Square/ReadVariableOp�*kernel/Regularizer_5/Square/ReadVariableOp�*kernel/Regularizer_6/Square/ReadVariableOp�*kernel/Regularizer_7/Square/ReadVariableOp�
conv3d/StatefulPartitionedCallStatefulPartitionedCallinput_1conv3d_11086conv3d_11088*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������22*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *J
fERC
A__inference_conv3d_layer_call_and_return_conditional_losses_10479�
 conv3d_1/StatefulPartitionedCallStatefulPartitionedCall'conv3d/StatefulPartitionedCall:output:0conv3d_1_11091conv3d_1_11093*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������22*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *L
fGRE
C__inference_conv3d_1_layer_call_and_return_conditional_losses_10502�
max_pooling3d/PartitionedCallPartitionedCall)conv3d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������
* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8� *Q
fLRJ
H__inference_max_pooling3d_layer_call_and_return_conditional_losses_10416�
 conv3d_2/StatefulPartitionedCallStatefulPartitionedCall&max_pooling3d/PartitionedCall:output:0conv3d_2_11097conv3d_2_11099*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������
 *$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *L
fGRE
C__inference_conv3d_2_layer_call_and_return_conditional_losses_10526�
 conv3d_3/StatefulPartitionedCallStatefulPartitionedCall)conv3d_2/StatefulPartitionedCall:output:0conv3d_3_11102conv3d_3_11104*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������
 *$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *L
fGRE
C__inference_conv3d_3_layer_call_and_return_conditional_losses_10549�
max_pooling3d_1/PartitionedCallPartitionedCall)conv3d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:��������� * 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8� *S
fNRL
J__inference_max_pooling3d_1_layer_call_and_return_conditional_losses_10428�
 conv3d_4/StatefulPartitionedCallStatefulPartitionedCall(max_pooling3d_1/PartitionedCall:output:0conv3d_4_11108conv3d_4_11110*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������@*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *L
fGRE
C__inference_conv3d_4_layer_call_and_return_conditional_losses_10573�
 conv3d_5/StatefulPartitionedCallStatefulPartitionedCall)conv3d_4/StatefulPartitionedCall:output:0conv3d_5_11113conv3d_5_11115*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������@*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *L
fGRE
C__inference_conv3d_5_layer_call_and_return_conditional_losses_10596�
max_pooling3d_2/PartitionedCallPartitionedCall)conv3d_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������@* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8� *S
fNRL
J__inference_max_pooling3d_2_layer_call_and_return_conditional_losses_10440�
 conv3d_6/StatefulPartitionedCallStatefulPartitionedCall(max_pooling3d_2/PartitionedCall:output:0conv3d_6_11119conv3d_6_11121*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :����������*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *L
fGRE
C__inference_conv3d_6_layer_call_and_return_conditional_losses_10620�
 conv3d_7/StatefulPartitionedCallStatefulPartitionedCall)conv3d_6/StatefulPartitionedCall:output:0conv3d_7_11124conv3d_7_11126*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :����������*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *L
fGRE
C__inference_conv3d_7_layer_call_and_return_conditional_losses_10643�
max_pooling3d_3/PartitionedCallPartitionedCall)conv3d_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :����������* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8� *S
fNRL
J__inference_max_pooling3d_3_layer_call_and_return_conditional_losses_10452�
flatten/PartitionedCallPartitionedCall(max_pooling3d_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������	* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8� *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_10656�
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_11131dense_11133*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_10668�
(kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv3d_11086**
_output_shapes
:*
dtype0�
kernel/Regularizer/SquareSquare0kernel/Regularizer/Square/ReadVariableOp:value:0*
T0**
_output_shapes
:u
kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
kernel/Regularizer/SumSumkernel/Regularizer/Square:y:0!kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: ]
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;�
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
*kernel/Regularizer_1/Square/ReadVariableOpReadVariableOpconv3d_1_11091**
_output_shapes
:*
dtype0�
kernel/Regularizer_1/SquareSquare2kernel/Regularizer_1/Square/ReadVariableOp:value:0*
T0**
_output_shapes
:w
kernel/Regularizer_1/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
kernel/Regularizer_1/SumSumkernel/Regularizer_1/Square:y:0#kernel/Regularizer_1/Const:output:0*
T0*
_output_shapes
: _
kernel/Regularizer_1/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;�
kernel/Regularizer_1/mulMul#kernel/Regularizer_1/mul/x:output:0!kernel/Regularizer_1/Sum:output:0*
T0*
_output_shapes
: �
*kernel/Regularizer_2/Square/ReadVariableOpReadVariableOpconv3d_2_11097**
_output_shapes
: *
dtype0�
kernel/Regularizer_2/SquareSquare2kernel/Regularizer_2/Square/ReadVariableOp:value:0*
T0**
_output_shapes
: w
kernel/Regularizer_2/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
kernel/Regularizer_2/SumSumkernel/Regularizer_2/Square:y:0#kernel/Regularizer_2/Const:output:0*
T0*
_output_shapes
: _
kernel/Regularizer_2/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;�
kernel/Regularizer_2/mulMul#kernel/Regularizer_2/mul/x:output:0!kernel/Regularizer_2/Sum:output:0*
T0*
_output_shapes
: �
*kernel/Regularizer_3/Square/ReadVariableOpReadVariableOpconv3d_3_11102**
_output_shapes
:  *
dtype0�
kernel/Regularizer_3/SquareSquare2kernel/Regularizer_3/Square/ReadVariableOp:value:0*
T0**
_output_shapes
:  w
kernel/Regularizer_3/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
kernel/Regularizer_3/SumSumkernel/Regularizer_3/Square:y:0#kernel/Regularizer_3/Const:output:0*
T0*
_output_shapes
: _
kernel/Regularizer_3/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;�
kernel/Regularizer_3/mulMul#kernel/Regularizer_3/mul/x:output:0!kernel/Regularizer_3/Sum:output:0*
T0*
_output_shapes
: �
*kernel/Regularizer_4/Square/ReadVariableOpReadVariableOpconv3d_4_11108**
_output_shapes
: @*
dtype0�
kernel/Regularizer_4/SquareSquare2kernel/Regularizer_4/Square/ReadVariableOp:value:0*
T0**
_output_shapes
: @w
kernel/Regularizer_4/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
kernel/Regularizer_4/SumSumkernel/Regularizer_4/Square:y:0#kernel/Regularizer_4/Const:output:0*
T0*
_output_shapes
: _
kernel/Regularizer_4/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;�
kernel/Regularizer_4/mulMul#kernel/Regularizer_4/mul/x:output:0!kernel/Regularizer_4/Sum:output:0*
T0*
_output_shapes
: �
*kernel/Regularizer_5/Square/ReadVariableOpReadVariableOpconv3d_5_11113**
_output_shapes
:@@*
dtype0�
kernel/Regularizer_5/SquareSquare2kernel/Regularizer_5/Square/ReadVariableOp:value:0*
T0**
_output_shapes
:@@w
kernel/Regularizer_5/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
kernel/Regularizer_5/SumSumkernel/Regularizer_5/Square:y:0#kernel/Regularizer_5/Const:output:0*
T0*
_output_shapes
: _
kernel/Regularizer_5/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;�
kernel/Regularizer_5/mulMul#kernel/Regularizer_5/mul/x:output:0!kernel/Regularizer_5/Sum:output:0*
T0*
_output_shapes
: �
*kernel/Regularizer_6/Square/ReadVariableOpReadVariableOpconv3d_6_11119*+
_output_shapes
:@�*
dtype0�
kernel/Regularizer_6/SquareSquare2kernel/Regularizer_6/Square/ReadVariableOp:value:0*
T0*+
_output_shapes
:@�w
kernel/Regularizer_6/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
kernel/Regularizer_6/SumSumkernel/Regularizer_6/Square:y:0#kernel/Regularizer_6/Const:output:0*
T0*
_output_shapes
: _
kernel/Regularizer_6/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;�
kernel/Regularizer_6/mulMul#kernel/Regularizer_6/mul/x:output:0!kernel/Regularizer_6/Sum:output:0*
T0*
_output_shapes
: �
*kernel/Regularizer_7/Square/ReadVariableOpReadVariableOpconv3d_7_11124*,
_output_shapes
:��*
dtype0�
kernel/Regularizer_7/SquareSquare2kernel/Regularizer_7/Square/ReadVariableOp:value:0*
T0*,
_output_shapes
:��w
kernel/Regularizer_7/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
kernel/Regularizer_7/SumSumkernel/Regularizer_7/Square:y:0#kernel/Regularizer_7/Const:output:0*
T0*
_output_shapes
: _
kernel/Regularizer_7/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;�
kernel/Regularizer_7/mulMul#kernel/Regularizer_7/mul/x:output:0!kernel/Regularizer_7/Sum:output:0*
T0*
_output_shapes
: u
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^conv3d/StatefulPartitionedCall!^conv3d_1/StatefulPartitionedCall!^conv3d_2/StatefulPartitionedCall!^conv3d_3/StatefulPartitionedCall!^conv3d_4/StatefulPartitionedCall!^conv3d_5/StatefulPartitionedCall!^conv3d_6/StatefulPartitionedCall!^conv3d_7/StatefulPartitionedCall^dense/StatefulPartitionedCall)^kernel/Regularizer/Square/ReadVariableOp+^kernel/Regularizer_1/Square/ReadVariableOp+^kernel/Regularizer_2/Square/ReadVariableOp+^kernel/Regularizer_3/Square/ReadVariableOp+^kernel/Regularizer_4/Square/ReadVariableOp+^kernel/Regularizer_5/Square/ReadVariableOp+^kernel/Regularizer_6/Square/ReadVariableOp+^kernel/Regularizer_7/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:���������22: : : : : : : : : : : : : : : : : : 2@
conv3d/StatefulPartitionedCallconv3d/StatefulPartitionedCall2D
 conv3d_1/StatefulPartitionedCall conv3d_1/StatefulPartitionedCall2D
 conv3d_2/StatefulPartitionedCall conv3d_2/StatefulPartitionedCall2D
 conv3d_3/StatefulPartitionedCall conv3d_3/StatefulPartitionedCall2D
 conv3d_4/StatefulPartitionedCall conv3d_4/StatefulPartitionedCall2D
 conv3d_5/StatefulPartitionedCall conv3d_5/StatefulPartitionedCall2D
 conv3d_6/StatefulPartitionedCall conv3d_6/StatefulPartitionedCall2D
 conv3d_7/StatefulPartitionedCall conv3d_7/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2T
(kernel/Regularizer/Square/ReadVariableOp(kernel/Regularizer/Square/ReadVariableOp2X
*kernel/Regularizer_1/Square/ReadVariableOp*kernel/Regularizer_1/Square/ReadVariableOp2X
*kernel/Regularizer_2/Square/ReadVariableOp*kernel/Regularizer_2/Square/ReadVariableOp2X
*kernel/Regularizer_3/Square/ReadVariableOp*kernel/Regularizer_3/Square/ReadVariableOp2X
*kernel/Regularizer_4/Square/ReadVariableOp*kernel/Regularizer_4/Square/ReadVariableOp2X
*kernel/Regularizer_5/Square/ReadVariableOp*kernel/Regularizer_5/Square/ReadVariableOp2X
*kernel/Regularizer_6/Square/ReadVariableOp*kernel/Regularizer_6/Square/ReadVariableOp2X
*kernel/Regularizer_7/Square/ReadVariableOp*kernel/Regularizer_7/Square/ReadVariableOp:\ X
3
_output_shapes!
:���������22
!
_user_specified_name	input_1
�
�
(__inference_conv3d_3_layer_call_fn_11869

inputs%
unknown:  
	unknown_0: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������
 *$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *L
fGRE
C__inference_conv3d_3_layer_call_and_return_conditional_losses_10549{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:���������
 `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������
 : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:���������
 
 
_user_specified_nameinputs
�
�
C__inference_conv3d_7_layer_call_and_return_conditional_losses_10643

inputs>
conv3d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv3D/ReadVariableOp�(kernel/Regularizer/Square/ReadVariableOp�
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource*,
_output_shapes
:��*
dtype0�
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :����������*
paddingSAME*
strides	
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :����������]
ReluReluBiasAdd:output:0*
T0*4
_output_shapes"
 :�����������
(kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv3d_readvariableop_resource*,
_output_shapes
:��*
dtype0�
kernel/Regularizer/SquareSquare0kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*,
_output_shapes
:��u
kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
kernel/Regularizer/SumSumkernel/Regularizer/Square:y:0!kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: ]
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;�
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: n
IdentityIdentityRelu:activations:0^NoOp*
T0*4
_output_shapes"
 :�����������
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp)^kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp2T
(kernel/Regularizer/Square/ReadVariableOp(kernel/Regularizer/Square/ReadVariableOp:\ X
4
_output_shapes"
 :����������
 
_user_specified_nameinputs
�
^
B__inference_flatten_layer_call_and_return_conditional_losses_10656

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"�����  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������	Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������:\ X
4
_output_shapes"
 :����������
 
_user_specified_nameinputs
�z
�
@__inference_model_layer_call_and_return_conditional_losses_11287
input_1*
conv3d_11188:
conv3d_11190:,
conv3d_1_11193:
conv3d_1_11195:,
conv3d_2_11199: 
conv3d_2_11201: ,
conv3d_3_11204:  
conv3d_3_11206: ,
conv3d_4_11210: @
conv3d_4_11212:@,
conv3d_5_11215:@@
conv3d_5_11217:@-
conv3d_6_11221:@�
conv3d_6_11223:	�.
conv3d_7_11226:��
conv3d_7_11228:	�
dense_11233:	�	
dense_11235:
identity��conv3d/StatefulPartitionedCall� conv3d_1/StatefulPartitionedCall� conv3d_2/StatefulPartitionedCall� conv3d_3/StatefulPartitionedCall� conv3d_4/StatefulPartitionedCall� conv3d_5/StatefulPartitionedCall� conv3d_6/StatefulPartitionedCall� conv3d_7/StatefulPartitionedCall�dense/StatefulPartitionedCall�(kernel/Regularizer/Square/ReadVariableOp�*kernel/Regularizer_1/Square/ReadVariableOp�*kernel/Regularizer_2/Square/ReadVariableOp�*kernel/Regularizer_3/Square/ReadVariableOp�*kernel/Regularizer_4/Square/ReadVariableOp�*kernel/Regularizer_5/Square/ReadVariableOp�*kernel/Regularizer_6/Square/ReadVariableOp�*kernel/Regularizer_7/Square/ReadVariableOp�
conv3d/StatefulPartitionedCallStatefulPartitionedCallinput_1conv3d_11188conv3d_11190*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������22*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *J
fERC
A__inference_conv3d_layer_call_and_return_conditional_losses_10479�
 conv3d_1/StatefulPartitionedCallStatefulPartitionedCall'conv3d/StatefulPartitionedCall:output:0conv3d_1_11193conv3d_1_11195*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������22*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *L
fGRE
C__inference_conv3d_1_layer_call_and_return_conditional_losses_10502�
max_pooling3d/PartitionedCallPartitionedCall)conv3d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������
* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8� *Q
fLRJ
H__inference_max_pooling3d_layer_call_and_return_conditional_losses_10416�
 conv3d_2/StatefulPartitionedCallStatefulPartitionedCall&max_pooling3d/PartitionedCall:output:0conv3d_2_11199conv3d_2_11201*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������
 *$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *L
fGRE
C__inference_conv3d_2_layer_call_and_return_conditional_losses_10526�
 conv3d_3/StatefulPartitionedCallStatefulPartitionedCall)conv3d_2/StatefulPartitionedCall:output:0conv3d_3_11204conv3d_3_11206*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������
 *$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *L
fGRE
C__inference_conv3d_3_layer_call_and_return_conditional_losses_10549�
max_pooling3d_1/PartitionedCallPartitionedCall)conv3d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:��������� * 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8� *S
fNRL
J__inference_max_pooling3d_1_layer_call_and_return_conditional_losses_10428�
 conv3d_4/StatefulPartitionedCallStatefulPartitionedCall(max_pooling3d_1/PartitionedCall:output:0conv3d_4_11210conv3d_4_11212*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������@*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *L
fGRE
C__inference_conv3d_4_layer_call_and_return_conditional_losses_10573�
 conv3d_5/StatefulPartitionedCallStatefulPartitionedCall)conv3d_4/StatefulPartitionedCall:output:0conv3d_5_11215conv3d_5_11217*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������@*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *L
fGRE
C__inference_conv3d_5_layer_call_and_return_conditional_losses_10596�
max_pooling3d_2/PartitionedCallPartitionedCall)conv3d_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������@* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8� *S
fNRL
J__inference_max_pooling3d_2_layer_call_and_return_conditional_losses_10440�
 conv3d_6/StatefulPartitionedCallStatefulPartitionedCall(max_pooling3d_2/PartitionedCall:output:0conv3d_6_11221conv3d_6_11223*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :����������*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *L
fGRE
C__inference_conv3d_6_layer_call_and_return_conditional_losses_10620�
 conv3d_7/StatefulPartitionedCallStatefulPartitionedCall)conv3d_6/StatefulPartitionedCall:output:0conv3d_7_11226conv3d_7_11228*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :����������*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *L
fGRE
C__inference_conv3d_7_layer_call_and_return_conditional_losses_10643�
max_pooling3d_3/PartitionedCallPartitionedCall)conv3d_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :����������* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8� *S
fNRL
J__inference_max_pooling3d_3_layer_call_and_return_conditional_losses_10452�
flatten/PartitionedCallPartitionedCall(max_pooling3d_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������	* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8� *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_10656�
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_11233dense_11235*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_10668�
(kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv3d_11188**
_output_shapes
:*
dtype0�
kernel/Regularizer/SquareSquare0kernel/Regularizer/Square/ReadVariableOp:value:0*
T0**
_output_shapes
:u
kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
kernel/Regularizer/SumSumkernel/Regularizer/Square:y:0!kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: ]
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;�
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
*kernel/Regularizer_1/Square/ReadVariableOpReadVariableOpconv3d_1_11193**
_output_shapes
:*
dtype0�
kernel/Regularizer_1/SquareSquare2kernel/Regularizer_1/Square/ReadVariableOp:value:0*
T0**
_output_shapes
:w
kernel/Regularizer_1/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
kernel/Regularizer_1/SumSumkernel/Regularizer_1/Square:y:0#kernel/Regularizer_1/Const:output:0*
T0*
_output_shapes
: _
kernel/Regularizer_1/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;�
kernel/Regularizer_1/mulMul#kernel/Regularizer_1/mul/x:output:0!kernel/Regularizer_1/Sum:output:0*
T0*
_output_shapes
: �
*kernel/Regularizer_2/Square/ReadVariableOpReadVariableOpconv3d_2_11199**
_output_shapes
: *
dtype0�
kernel/Regularizer_2/SquareSquare2kernel/Regularizer_2/Square/ReadVariableOp:value:0*
T0**
_output_shapes
: w
kernel/Regularizer_2/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
kernel/Regularizer_2/SumSumkernel/Regularizer_2/Square:y:0#kernel/Regularizer_2/Const:output:0*
T0*
_output_shapes
: _
kernel/Regularizer_2/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;�
kernel/Regularizer_2/mulMul#kernel/Regularizer_2/mul/x:output:0!kernel/Regularizer_2/Sum:output:0*
T0*
_output_shapes
: �
*kernel/Regularizer_3/Square/ReadVariableOpReadVariableOpconv3d_3_11204**
_output_shapes
:  *
dtype0�
kernel/Regularizer_3/SquareSquare2kernel/Regularizer_3/Square/ReadVariableOp:value:0*
T0**
_output_shapes
:  w
kernel/Regularizer_3/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
kernel/Regularizer_3/SumSumkernel/Regularizer_3/Square:y:0#kernel/Regularizer_3/Const:output:0*
T0*
_output_shapes
: _
kernel/Regularizer_3/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;�
kernel/Regularizer_3/mulMul#kernel/Regularizer_3/mul/x:output:0!kernel/Regularizer_3/Sum:output:0*
T0*
_output_shapes
: �
*kernel/Regularizer_4/Square/ReadVariableOpReadVariableOpconv3d_4_11210**
_output_shapes
: @*
dtype0�
kernel/Regularizer_4/SquareSquare2kernel/Regularizer_4/Square/ReadVariableOp:value:0*
T0**
_output_shapes
: @w
kernel/Regularizer_4/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
kernel/Regularizer_4/SumSumkernel/Regularizer_4/Square:y:0#kernel/Regularizer_4/Const:output:0*
T0*
_output_shapes
: _
kernel/Regularizer_4/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;�
kernel/Regularizer_4/mulMul#kernel/Regularizer_4/mul/x:output:0!kernel/Regularizer_4/Sum:output:0*
T0*
_output_shapes
: �
*kernel/Regularizer_5/Square/ReadVariableOpReadVariableOpconv3d_5_11215**
_output_shapes
:@@*
dtype0�
kernel/Regularizer_5/SquareSquare2kernel/Regularizer_5/Square/ReadVariableOp:value:0*
T0**
_output_shapes
:@@w
kernel/Regularizer_5/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
kernel/Regularizer_5/SumSumkernel/Regularizer_5/Square:y:0#kernel/Regularizer_5/Const:output:0*
T0*
_output_shapes
: _
kernel/Regularizer_5/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;�
kernel/Regularizer_5/mulMul#kernel/Regularizer_5/mul/x:output:0!kernel/Regularizer_5/Sum:output:0*
T0*
_output_shapes
: �
*kernel/Regularizer_6/Square/ReadVariableOpReadVariableOpconv3d_6_11221*+
_output_shapes
:@�*
dtype0�
kernel/Regularizer_6/SquareSquare2kernel/Regularizer_6/Square/ReadVariableOp:value:0*
T0*+
_output_shapes
:@�w
kernel/Regularizer_6/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
kernel/Regularizer_6/SumSumkernel/Regularizer_6/Square:y:0#kernel/Regularizer_6/Const:output:0*
T0*
_output_shapes
: _
kernel/Regularizer_6/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;�
kernel/Regularizer_6/mulMul#kernel/Regularizer_6/mul/x:output:0!kernel/Regularizer_6/Sum:output:0*
T0*
_output_shapes
: �
*kernel/Regularizer_7/Square/ReadVariableOpReadVariableOpconv3d_7_11226*,
_output_shapes
:��*
dtype0�
kernel/Regularizer_7/SquareSquare2kernel/Regularizer_7/Square/ReadVariableOp:value:0*
T0*,
_output_shapes
:��w
kernel/Regularizer_7/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
kernel/Regularizer_7/SumSumkernel/Regularizer_7/Square:y:0#kernel/Regularizer_7/Const:output:0*
T0*
_output_shapes
: _
kernel/Regularizer_7/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;�
kernel/Regularizer_7/mulMul#kernel/Regularizer_7/mul/x:output:0!kernel/Regularizer_7/Sum:output:0*
T0*
_output_shapes
: u
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^conv3d/StatefulPartitionedCall!^conv3d_1/StatefulPartitionedCall!^conv3d_2/StatefulPartitionedCall!^conv3d_3/StatefulPartitionedCall!^conv3d_4/StatefulPartitionedCall!^conv3d_5/StatefulPartitionedCall!^conv3d_6/StatefulPartitionedCall!^conv3d_7/StatefulPartitionedCall^dense/StatefulPartitionedCall)^kernel/Regularizer/Square/ReadVariableOp+^kernel/Regularizer_1/Square/ReadVariableOp+^kernel/Regularizer_2/Square/ReadVariableOp+^kernel/Regularizer_3/Square/ReadVariableOp+^kernel/Regularizer_4/Square/ReadVariableOp+^kernel/Regularizer_5/Square/ReadVariableOp+^kernel/Regularizer_6/Square/ReadVariableOp+^kernel/Regularizer_7/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:���������22: : : : : : : : : : : : : : : : : : 2@
conv3d/StatefulPartitionedCallconv3d/StatefulPartitionedCall2D
 conv3d_1/StatefulPartitionedCall conv3d_1/StatefulPartitionedCall2D
 conv3d_2/StatefulPartitionedCall conv3d_2/StatefulPartitionedCall2D
 conv3d_3/StatefulPartitionedCall conv3d_3/StatefulPartitionedCall2D
 conv3d_4/StatefulPartitionedCall conv3d_4/StatefulPartitionedCall2D
 conv3d_5/StatefulPartitionedCall conv3d_5/StatefulPartitionedCall2D
 conv3d_6/StatefulPartitionedCall conv3d_6/StatefulPartitionedCall2D
 conv3d_7/StatefulPartitionedCall conv3d_7/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2T
(kernel/Regularizer/Square/ReadVariableOp(kernel/Regularizer/Square/ReadVariableOp2X
*kernel/Regularizer_1/Square/ReadVariableOp*kernel/Regularizer_1/Square/ReadVariableOp2X
*kernel/Regularizer_2/Square/ReadVariableOp*kernel/Regularizer_2/Square/ReadVariableOp2X
*kernel/Regularizer_3/Square/ReadVariableOp*kernel/Regularizer_3/Square/ReadVariableOp2X
*kernel/Regularizer_4/Square/ReadVariableOp*kernel/Regularizer_4/Square/ReadVariableOp2X
*kernel/Regularizer_5/Square/ReadVariableOp*kernel/Regularizer_5/Square/ReadVariableOp2X
*kernel/Regularizer_6/Square/ReadVariableOp*kernel/Regularizer_6/Square/ReadVariableOp2X
*kernel/Regularizer_7/Square/ReadVariableOp*kernel/Regularizer_7/Square/ReadVariableOp:\ X
3
_output_shapes!
:���������22
!
_user_specified_name	input_1
�
�
C__inference_conv3d_1_layer_call_and_return_conditional_losses_11824

inputs<
conv3d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv3D/ReadVariableOp�(kernel/Regularizer/Square/ReadVariableOp�
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������22*
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������22\
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:���������22�
(kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype0�
kernel/Regularizer/SquareSquare0kernel/Regularizer/Square/ReadVariableOp:value:0*
T0**
_output_shapes
:u
kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
kernel/Regularizer/SumSumkernel/Regularizer/Square:y:0!kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: ]
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;�
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:���������22�
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp)^kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������22: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp2T
(kernel/Regularizer/Square/ReadVariableOp(kernel/Regularizer/Square/ReadVariableOp:[ W
3
_output_shapes!
:���������22
 
_user_specified_nameinputs
�
f
J__inference_max_pooling3d_2_layer_call_and_return_conditional_losses_10440

inputs
identity�
	MaxPool3D	MaxPool3Dinputs*
T0*W
_output_shapesE
C:A���������������������������������������������*
ksize	
*
paddingVALID*
strides	
�
IdentityIdentityMaxPool3D:output:0*
T0*W
_output_shapesE
C:A���������������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A���������������������������������������������: {
W
_output_shapesE
C:A���������������������������������������������
 
_user_specified_nameinputs
�
C
'__inference_flatten_layer_call_fn_12025

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������	* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8� *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_10656a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������:\ X
4
_output_shapes"
 :����������
 
_user_specified_nameinputs
�
�
C__inference_conv3d_5_layer_call_and_return_conditional_losses_10596

inputs<
conv3d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�Conv3D/ReadVariableOp�(kernel/Regularizer/Square/ReadVariableOp�
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:@@*
dtype0�
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������@*
paddingSAME*
strides	
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������@\
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:���������@�
(kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:@@*
dtype0�
kernel/Regularizer/SquareSquare0kernel/Regularizer/Square/ReadVariableOp:value:0*
T0**
_output_shapes
:@@u
kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
kernel/Regularizer/SumSumkernel/Regularizer/Square:y:0!kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: ]
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;�
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:���������@�
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp)^kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp2T
(kernel/Regularizer/Square/ReadVariableOp(kernel/Regularizer/Square/ReadVariableOp:[ W
3
_output_shapes!
:���������@
 
_user_specified_nameinputs
�
f
J__inference_max_pooling3d_3_layer_call_and_return_conditional_losses_10452

inputs
identity�
	MaxPool3D	MaxPool3Dinputs*
T0*W
_output_shapesE
C:A���������������������������������������������*
ksize	
*
paddingVALID*
strides	
�
IdentityIdentityMaxPool3D:output:0*
T0*W
_output_shapesE
C:A���������������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A���������������������������������������������: {
W
_output_shapesE
C:A���������������������������������������������
 
_user_specified_nameinputs
�
I
-__inference_max_pooling3d_layer_call_fn_11829

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *W
_output_shapesE
C:A���������������������������������������������* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8� *Q
fLRJ
H__inference_max_pooling3d_layer_call_and_return_conditional_losses_10416�
IdentityIdentityPartitionedCall:output:0*
T0*W
_output_shapesE
C:A���������������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:A���������������������������������������������: {
W
_output_shapesE
C:A���������������������������������������������
 
_user_specified_nameinputs
�

�
__inference_loss_fn_1_12072O
1kernel_regularizer_square_readvariableop_resource:
identity��(kernel/Regularizer/Square/ReadVariableOp�
(kernel/Regularizer/Square/ReadVariableOpReadVariableOp1kernel_regularizer_square_readvariableop_resource**
_output_shapes
:*
dtype0�
kernel/Regularizer/SquareSquare0kernel/Regularizer/Square/ReadVariableOp:value:0*
T0**
_output_shapes
:u
kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
kernel/Regularizer/SumSumkernel/Regularizer/Square:y:0!kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: ]
kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;�
kernel/Regularizer/mulMul!kernel/Regularizer/mul/x:output:0kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: X
IdentityIdentitykernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: q
NoOpNoOp)^kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2T
(kernel/Regularizer/Square/ReadVariableOp(kernel/Regularizer/Square/ReadVariableOp"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
G
input_1<
serving_default_input_1:0���������229
dense0
StatefulPartitionedCall:0���������tensorflow/serving/predict:��
�
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer-6
layer_with_weights-4
layer-7
	layer_with_weights-5
	layer-8

layer-9
layer_with_weights-6
layer-10
layer_with_weights-7
layer-11
layer-12
layer-13
layer_with_weights-8
layer-14
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer
loss

signatures"
_tf_keras_network
D
#_self_saveable_object_factories"
_tf_keras_input_layer
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
* &call_and_return_all_conditional_losses

!kernel
"bias
##_self_saveable_object_factories
 $_jit_compiled_convolution_op"
_tf_keras_layer
�
%	variables
&trainable_variables
'regularization_losses
(	keras_api
)__call__
**&call_and_return_all_conditional_losses

+kernel
,bias
#-_self_saveable_object_factories
 ._jit_compiled_convolution_op"
_tf_keras_layer
�
/	variables
0trainable_variables
1regularization_losses
2	keras_api
3__call__
*4&call_and_return_all_conditional_losses
#5_self_saveable_object_factories"
_tf_keras_layer
�
6	variables
7trainable_variables
8regularization_losses
9	keras_api
:__call__
*;&call_and_return_all_conditional_losses

<kernel
=bias
#>_self_saveable_object_factories
 ?_jit_compiled_convolution_op"
_tf_keras_layer
�
@	variables
Atrainable_variables
Bregularization_losses
C	keras_api
D__call__
*E&call_and_return_all_conditional_losses

Fkernel
Gbias
#H_self_saveable_object_factories
 I_jit_compiled_convolution_op"
_tf_keras_layer
�
J	variables
Ktrainable_variables
Lregularization_losses
M	keras_api
N__call__
*O&call_and_return_all_conditional_losses
#P_self_saveable_object_factories"
_tf_keras_layer
�
Q	variables
Rtrainable_variables
Sregularization_losses
T	keras_api
U__call__
*V&call_and_return_all_conditional_losses

Wkernel
Xbias
#Y_self_saveable_object_factories
 Z_jit_compiled_convolution_op"
_tf_keras_layer
�
[	variables
\trainable_variables
]regularization_losses
^	keras_api
___call__
*`&call_and_return_all_conditional_losses

akernel
bbias
#c_self_saveable_object_factories
 d_jit_compiled_convolution_op"
_tf_keras_layer
�
e	variables
ftrainable_variables
gregularization_losses
h	keras_api
i__call__
*j&call_and_return_all_conditional_losses
#k_self_saveable_object_factories"
_tf_keras_layer
�
l	variables
mtrainable_variables
nregularization_losses
o	keras_api
p__call__
*q&call_and_return_all_conditional_losses

rkernel
sbias
#t_self_saveable_object_factories
 u_jit_compiled_convolution_op"
_tf_keras_layer
�
v	variables
wtrainable_variables
xregularization_losses
y	keras_api
z__call__
*{&call_and_return_all_conditional_losses

|kernel
}bias
#~_self_saveable_object_factories
 _jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
$�_self_saveable_object_factories"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
$�_self_saveable_object_factories"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias"
_tf_keras_layer
�
!0
"1
+2
,3
<4
=5
F6
G7
W8
X9
a10
b11
r12
s13
|14
}15
�16
�17"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
`
�0
�1
�2
�3
�4
�5
�6
�7"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_1
�trace_2
�trace_32�
%__inference_model_layer_call_fn_10762
%__inference_model_layer_call_fn_11491
%__inference_model_layer_call_fn_11532
%__inference_model_layer_call_fn_11083�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 z�trace_0z�trace_1z�trace_2z�trace_3
�
�trace_0
�trace_1
�trace_2
�trace_32�
@__inference_model_layer_call_and_return_conditional_losses_11652
@__inference_model_layer_call_and_return_conditional_losses_11772
@__inference_model_layer_call_and_return_conditional_losses_11185
@__inference_model_layer_call_and_return_conditional_losses_11287�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 z�trace_0z�trace_1z�trace_2z�trace_3
�B�
 __inference__wrapped_model_10407input_1"�
���
FullArgSpec
args� 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�
	�iter
�beta_1
�beta_2

�decay
�learning_rate	�m�	�m�	�v�	�v�"
	optimizer
�
�trace_0
�trace_12�
__inference_loss_fn_11350
__inference_loss_fn_11359�
���
FullArgSpec
args�
jy_true
jy_pred
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
-
�serving_default"
signature_map
 "
trackable_dict_wrapper
.
!0
"1"
trackable_list_wrapper
 "
trackable_list_wrapper
(
�0"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
__call__
* &call_and_return_all_conditional_losses
& "call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
&__inference_conv3d_layer_call_fn_11781�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
A__inference_conv3d_layer_call_and_return_conditional_losses_11798�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
+:)2conv3d/kernel
:2conv3d/bias
 "
trackable_dict_wrapper
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
.
+0
,1"
trackable_list_wrapper
 "
trackable_list_wrapper
(
�0"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
%	variables
&trainable_variables
'regularization_losses
)__call__
**&call_and_return_all_conditional_losses
&*"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
(__inference_conv3d_1_layer_call_fn_11807�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
C__inference_conv3d_1_layer_call_and_return_conditional_losses_11824�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
-:+2conv3d_1/kernel
:2conv3d_1/bias
 "
trackable_dict_wrapper
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
/	variables
0trainable_variables
1regularization_losses
3__call__
*4&call_and_return_all_conditional_losses
&4"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
-__inference_max_pooling3d_layer_call_fn_11829�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
H__inference_max_pooling3d_layer_call_and_return_conditional_losses_11834�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 "
trackable_dict_wrapper
.
<0
=1"
trackable_list_wrapper
 "
trackable_list_wrapper
(
�0"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
6	variables
7trainable_variables
8regularization_losses
:__call__
*;&call_and_return_all_conditional_losses
&;"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
(__inference_conv3d_2_layer_call_fn_11843�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
C__inference_conv3d_2_layer_call_and_return_conditional_losses_11860�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
-:+ 2conv3d_2/kernel
: 2conv3d_2/bias
 "
trackable_dict_wrapper
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
.
F0
G1"
trackable_list_wrapper
 "
trackable_list_wrapper
(
�0"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
@	variables
Atrainable_variables
Bregularization_losses
D__call__
*E&call_and_return_all_conditional_losses
&E"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
(__inference_conv3d_3_layer_call_fn_11869�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
C__inference_conv3d_3_layer_call_and_return_conditional_losses_11886�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
-:+  2conv3d_3/kernel
: 2conv3d_3/bias
 "
trackable_dict_wrapper
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
J	variables
Ktrainable_variables
Lregularization_losses
N__call__
*O&call_and_return_all_conditional_losses
&O"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
/__inference_max_pooling3d_1_layer_call_fn_11891�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
J__inference_max_pooling3d_1_layer_call_and_return_conditional_losses_11896�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 "
trackable_dict_wrapper
.
W0
X1"
trackable_list_wrapper
 "
trackable_list_wrapper
(
�0"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
Q	variables
Rtrainable_variables
Sregularization_losses
U__call__
*V&call_and_return_all_conditional_losses
&V"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
(__inference_conv3d_4_layer_call_fn_11905�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
C__inference_conv3d_4_layer_call_and_return_conditional_losses_11922�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
-:+ @2conv3d_4/kernel
:@2conv3d_4/bias
 "
trackable_dict_wrapper
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
.
a0
b1"
trackable_list_wrapper
 "
trackable_list_wrapper
(
�0"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
[	variables
\trainable_variables
]regularization_losses
___call__
*`&call_and_return_all_conditional_losses
&`"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
(__inference_conv3d_5_layer_call_fn_11931�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
C__inference_conv3d_5_layer_call_and_return_conditional_losses_11948�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
-:+@@2conv3d_5/kernel
:@2conv3d_5/bias
 "
trackable_dict_wrapper
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
e	variables
ftrainable_variables
gregularization_losses
i__call__
*j&call_and_return_all_conditional_losses
&j"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
/__inference_max_pooling3d_2_layer_call_fn_11953�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
J__inference_max_pooling3d_2_layer_call_and_return_conditional_losses_11958�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 "
trackable_dict_wrapper
.
r0
s1"
trackable_list_wrapper
 "
trackable_list_wrapper
(
�0"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
l	variables
mtrainable_variables
nregularization_losses
p__call__
*q&call_and_return_all_conditional_losses
&q"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
(__inference_conv3d_6_layer_call_fn_11967�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
C__inference_conv3d_6_layer_call_and_return_conditional_losses_11984�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
.:,@�2conv3d_6/kernel
:�2conv3d_6/bias
 "
trackable_dict_wrapper
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
.
|0
}1"
trackable_list_wrapper
 "
trackable_list_wrapper
(
�0"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
v	variables
wtrainable_variables
xregularization_losses
z__call__
*{&call_and_return_all_conditional_losses
&{"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
(__inference_conv3d_7_layer_call_fn_11993�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
C__inference_conv3d_7_layer_call_and_return_conditional_losses_12010�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
/:-��2conv3d_7/kernel
:�2conv3d_7/bias
 "
trackable_dict_wrapper
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
/__inference_max_pooling3d_3_layer_call_fn_12015�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
J__inference_max_pooling3d_3_layer_call_and_return_conditional_losses_12020�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
'__inference_flatten_layer_call_fn_12025�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
B__inference_flatten_layer_call_and_return_conditional_losses_12031�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 "
trackable_dict_wrapper
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
%__inference_dense_layer_call_fn_12040�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
@__inference_dense_layer_call_and_return_conditional_losses_12050�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
:	�	2dense/kernel
:2
dense/bias
�
�trace_02�
__inference_loss_fn_0_12061�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� z�trace_0
�
�trace_02�
__inference_loss_fn_1_12072�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� z�trace_0
�
�trace_02�
__inference_loss_fn_2_12083�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� z�trace_0
�
�trace_02�
__inference_loss_fn_3_12094�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� z�trace_0
�
�trace_02�
__inference_loss_fn_4_12105�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� z�trace_0
�
�trace_02�
__inference_loss_fn_5_12116�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� z�trace_0
�
�trace_02�
__inference_loss_fn_6_12127�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� z�trace_0
�
�trace_02�
__inference_loss_fn_7_12138�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� z�trace_0
�
!0
"1
+2
,3
<4
=5
F6
G7
W8
X9
a10
b11
r12
s13
|14
}15"
trackable_list_wrapper
�
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
%__inference_model_layer_call_fn_10762input_1"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
%__inference_model_layer_call_fn_11491inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
%__inference_model_layer_call_fn_11532inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
%__inference_model_layer_call_fn_11083input_1"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
@__inference_model_layer_call_and_return_conditional_losses_11652inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
@__inference_model_layer_call_and_return_conditional_losses_11772inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
@__inference_model_layer_call_and_return_conditional_losses_11185input_1"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
@__inference_model_layer_call_and_return_conditional_losses_11287input_1"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
�B�
__inference_loss_fn_11350y_truey_pred"�
���
FullArgSpec
args�
jy_true
jy_pred
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
__inference_loss_fn_11359y_truey_pred"�
���
FullArgSpec
args�
jy_true
jy_pred
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
#__inference_signature_wrapper_11402input_1"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
.
!0
"1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
&__inference_conv3d_layer_call_fn_11781inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
A__inference_conv3d_layer_call_and_return_conditional_losses_11798inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
.
+0
,1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
(__inference_conv3d_1_layer_call_fn_11807inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_conv3d_1_layer_call_and_return_conditional_losses_11824inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
-__inference_max_pooling3d_layer_call_fn_11829inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
H__inference_max_pooling3d_layer_call_and_return_conditional_losses_11834inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
.
<0
=1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
(__inference_conv3d_2_layer_call_fn_11843inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_conv3d_2_layer_call_and_return_conditional_losses_11860inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
.
F0
G1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
(__inference_conv3d_3_layer_call_fn_11869inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_conv3d_3_layer_call_and_return_conditional_losses_11886inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
/__inference_max_pooling3d_1_layer_call_fn_11891inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
J__inference_max_pooling3d_1_layer_call_and_return_conditional_losses_11896inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
.
W0
X1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
(__inference_conv3d_4_layer_call_fn_11905inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_conv3d_4_layer_call_and_return_conditional_losses_11922inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
.
a0
b1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
(__inference_conv3d_5_layer_call_fn_11931inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_conv3d_5_layer_call_and_return_conditional_losses_11948inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
/__inference_max_pooling3d_2_layer_call_fn_11953inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
J__inference_max_pooling3d_2_layer_call_and_return_conditional_losses_11958inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
.
r0
s1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
(__inference_conv3d_6_layer_call_fn_11967inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_conv3d_6_layer_call_and_return_conditional_losses_11984inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
.
|0
}1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
(__inference_conv3d_7_layer_call_fn_11993inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_conv3d_7_layer_call_and_return_conditional_losses_12010inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
/__inference_max_pooling3d_3_layer_call_fn_12015inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
J__inference_max_pooling3d_3_layer_call_and_return_conditional_losses_12020inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
'__inference_flatten_layer_call_fn_12025inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
B__inference_flatten_layer_call_and_return_conditional_losses_12031inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
%__inference_dense_layer_call_fn_12040inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
@__inference_dense_layer_call_and_return_conditional_losses_12050inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
__inference_loss_fn_0_12061"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�B�
__inference_loss_fn_1_12072"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�B�
__inference_loss_fn_2_12083"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�B�
__inference_loss_fn_3_12094"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�B�
__inference_loss_fn_4_12105"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�B�
__inference_loss_fn_5_12116"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�B�
__inference_loss_fn_6_12127"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�B�
__inference_loss_fn_7_12138"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
R
�	variables
�	keras_api

�total

�count"
_tf_keras_metric
c
�	variables
�	keras_api

�total

�count
�
_fn_kwargs"
_tf_keras_metric
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
$:"	�	2Adam/dense/kernel/m
:2Adam/dense/bias/m
$:"	�	2Adam/dense/kernel/v
:2Adam/dense/bias/v�
 __inference__wrapped_model_10407�!"+,<=FGWXabrs|}��<�9
2�/
-�*
input_1���������22
� "-�*
(
dense�
dense����������
C__inference_conv3d_1_layer_call_and_return_conditional_losses_11824t+,;�8
1�.
,�)
inputs���������22
� "1�.
'�$
0���������22
� �
(__inference_conv3d_1_layer_call_fn_11807g+,;�8
1�.
,�)
inputs���������22
� "$�!���������22�
C__inference_conv3d_2_layer_call_and_return_conditional_losses_11860t<=;�8
1�.
,�)
inputs���������

� "1�.
'�$
0���������
 
� �
(__inference_conv3d_2_layer_call_fn_11843g<=;�8
1�.
,�)
inputs���������

� "$�!���������
 �
C__inference_conv3d_3_layer_call_and_return_conditional_losses_11886tFG;�8
1�.
,�)
inputs���������
 
� "1�.
'�$
0���������
 
� �
(__inference_conv3d_3_layer_call_fn_11869gFG;�8
1�.
,�)
inputs���������
 
� "$�!���������
 �
C__inference_conv3d_4_layer_call_and_return_conditional_losses_11922tWX;�8
1�.
,�)
inputs��������� 
� "1�.
'�$
0���������@
� �
(__inference_conv3d_4_layer_call_fn_11905gWX;�8
1�.
,�)
inputs��������� 
� "$�!���������@�
C__inference_conv3d_5_layer_call_and_return_conditional_losses_11948tab;�8
1�.
,�)
inputs���������@
� "1�.
'�$
0���������@
� �
(__inference_conv3d_5_layer_call_fn_11931gab;�8
1�.
,�)
inputs���������@
� "$�!���������@�
C__inference_conv3d_6_layer_call_and_return_conditional_losses_11984urs;�8
1�.
,�)
inputs���������@
� "2�/
(�%
0����������
� �
(__inference_conv3d_6_layer_call_fn_11967hrs;�8
1�.
,�)
inputs���������@
� "%�"�����������
C__inference_conv3d_7_layer_call_and_return_conditional_losses_12010v|}<�9
2�/
-�*
inputs����������
� "2�/
(�%
0����������
� �
(__inference_conv3d_7_layer_call_fn_11993i|}<�9
2�/
-�*
inputs����������
� "%�"�����������
A__inference_conv3d_layer_call_and_return_conditional_losses_11798t!";�8
1�.
,�)
inputs���������22
� "1�.
'�$
0���������22
� �
&__inference_conv3d_layer_call_fn_11781g!";�8
1�.
,�)
inputs���������22
� "$�!���������22�
@__inference_dense_layer_call_and_return_conditional_losses_12050_��0�-
&�#
!�
inputs����������	
� "%�"
�
0���������
� {
%__inference_dense_layer_call_fn_12040R��0�-
&�#
!�
inputs����������	
� "�����������
B__inference_flatten_layer_call_and_return_conditional_losses_12031f<�9
2�/
-�*
inputs����������
� "&�#
�
0����������	
� �
'__inference_flatten_layer_call_fn_12025Y<�9
2�/
-�*
inputs����������
� "�����������	:
__inference_loss_fn_0_12061!�

� 
� "� g
__inference_loss_fn_11350J?�<
5�2
�
y_true 
�
y_pred 
� "� y
__inference_loss_fn_11359\Q�N
G�D
 �
y_true���������
 �
y_pred���������
� "� :
__inference_loss_fn_1_12072+�

� 
� "� :
__inference_loss_fn_2_12083<�

� 
� "� :
__inference_loss_fn_3_12094F�

� 
� "� :
__inference_loss_fn_4_12105W�

� 
� "� :
__inference_loss_fn_5_12116a�

� 
� "� :
__inference_loss_fn_6_12127r�

� 
� "� :
__inference_loss_fn_7_12138|�

� 
� "� �
J__inference_max_pooling3d_1_layer_call_and_return_conditional_losses_11896�_�\
U�R
P�M
inputsA���������������������������������������������
� "U�R
K�H
0A���������������������������������������������
� �
/__inference_max_pooling3d_1_layer_call_fn_11891�_�\
U�R
P�M
inputsA���������������������������������������������
� "H�EA����������������������������������������������
J__inference_max_pooling3d_2_layer_call_and_return_conditional_losses_11958�_�\
U�R
P�M
inputsA���������������������������������������������
� "U�R
K�H
0A���������������������������������������������
� �
/__inference_max_pooling3d_2_layer_call_fn_11953�_�\
U�R
P�M
inputsA���������������������������������������������
� "H�EA����������������������������������������������
J__inference_max_pooling3d_3_layer_call_and_return_conditional_losses_12020�_�\
U�R
P�M
inputsA���������������������������������������������
� "U�R
K�H
0A���������������������������������������������
� �
/__inference_max_pooling3d_3_layer_call_fn_12015�_�\
U�R
P�M
inputsA���������������������������������������������
� "H�EA����������������������������������������������
H__inference_max_pooling3d_layer_call_and_return_conditional_losses_11834�_�\
U�R
P�M
inputsA���������������������������������������������
� "U�R
K�H
0A���������������������������������������������
� �
-__inference_max_pooling3d_layer_call_fn_11829�_�\
U�R
P�M
inputsA���������������������������������������������
� "H�EA����������������������������������������������
@__inference_model_layer_call_and_return_conditional_losses_11185�!"+,<=FGWXabrs|}��D�A
:�7
-�*
input_1���������22
p 

 
� "%�"
�
0���������
� �
@__inference_model_layer_call_and_return_conditional_losses_11287�!"+,<=FGWXabrs|}��D�A
:�7
-�*
input_1���������22
p

 
� "%�"
�
0���������
� �
@__inference_model_layer_call_and_return_conditional_losses_11652�!"+,<=FGWXabrs|}��C�@
9�6
,�)
inputs���������22
p 

 
� "%�"
�
0���������
� �
@__inference_model_layer_call_and_return_conditional_losses_11772�!"+,<=FGWXabrs|}��C�@
9�6
,�)
inputs���������22
p

 
� "%�"
�
0���������
� �
%__inference_model_layer_call_fn_10762v!"+,<=FGWXabrs|}��D�A
:�7
-�*
input_1���������22
p 

 
� "�����������
%__inference_model_layer_call_fn_11083v!"+,<=FGWXabrs|}��D�A
:�7
-�*
input_1���������22
p

 
� "�����������
%__inference_model_layer_call_fn_11491u!"+,<=FGWXabrs|}��C�@
9�6
,�)
inputs���������22
p 

 
� "�����������
%__inference_model_layer_call_fn_11532u!"+,<=FGWXabrs|}��C�@
9�6
,�)
inputs���������22
p

 
� "�����������
#__inference_signature_wrapper_11402�!"+,<=FGWXabrs|}��G�D
� 
=�:
8
input_1-�*
input_1���������22"-�*
(
dense�
dense���������