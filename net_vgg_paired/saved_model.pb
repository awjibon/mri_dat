��)
��
D
AddV2
x"T
y"T
z"T"
Ttype:
2	��
E
AssignAddVariableOp
resource
value"dtype"
dtypetype�
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
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
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
9
DivNoNan
x"T
y"T
z"T"
Ttype:

2
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
>
Maximum
x"T
y"T
z"T"
Ttype:
2	
�
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(�
?
Mul
x"T
y"T
z"T"
Ttype:
2	�
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
8
Pow
x"T
y"T
z"T"
Ttype:
2
	
f
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx" 
Tidxtype0:
2
	
)
Rank

input"T

output"	
Ttype
@
ReadVariableOp
resource
value"dtype"
dtypetype�
@
RealDiv
x"T
y"T
z"T"
Ttype:
2	
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
-
Sqrt
x"T
y"T"
Ttype:

2
7
Square
x"T
y"T"
Ttype:
2	
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
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
<
Sub
x"T
y"T
z"T"
Ttype:
2	
�
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.9.02v2.9.0-rc2-42-g8a20d54a3c18߈%
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
shape:	�$*$
shared_nameAdam/dense/kernel/v
|
'Adam/dense/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/v*
_output_shapes
:	�$*
dtype0
�
Adam/conv3d_7/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*%
shared_nameAdam/conv3d_7/bias/v
z
(Adam/conv3d_7/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_7/bias/v*
_output_shapes	
:�*
dtype0
�
Adam/conv3d_7/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*!
shape:��*'
shared_nameAdam/conv3d_7/kernel/v
�
*Adam/conv3d_7/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_7/kernel/v*,
_output_shapes
:��*
dtype0
�
Adam/conv3d_6/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*%
shared_nameAdam/conv3d_6/bias/v
z
(Adam/conv3d_6/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_6/bias/v*
_output_shapes	
:�*
dtype0
�
Adam/conv3d_6/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*!
shape:��*'
shared_nameAdam/conv3d_6/kernel/v
�
*Adam/conv3d_6/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_6/kernel/v*,
_output_shapes
:��*
dtype0
�
Adam/conv3d_5/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*%
shared_nameAdam/conv3d_5/bias/v
z
(Adam/conv3d_5/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_5/bias/v*
_output_shapes	
:�*
dtype0
�
Adam/conv3d_5/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*!
shape:��*'
shared_nameAdam/conv3d_5/kernel/v
�
*Adam/conv3d_5/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_5/kernel/v*,
_output_shapes
:��*
dtype0
�
Adam/conv3d_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*%
shared_nameAdam/conv3d_4/bias/v
z
(Adam/conv3d_4/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_4/bias/v*
_output_shapes	
:�*
dtype0
�
Adam/conv3d_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*!
shape:��*'
shared_nameAdam/conv3d_4/kernel/v
�
*Adam/conv3d_4/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_4/kernel/v*,
_output_shapes
:��*
dtype0
�
Adam/conv3d_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*%
shared_nameAdam/conv3d_3/bias/v
z
(Adam/conv3d_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_3/bias/v*
_output_shapes	
:�*
dtype0
�
Adam/conv3d_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*!
shape:��*'
shared_nameAdam/conv3d_3/kernel/v
�
*Adam/conv3d_3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_3/kernel/v*,
_output_shapes
:��*
dtype0
�
Adam/conv3d_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*%
shared_nameAdam/conv3d_2/bias/v
z
(Adam/conv3d_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_2/bias/v*
_output_shapes	
:�*
dtype0
�
Adam/conv3d_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0* 
shape:@�*'
shared_nameAdam/conv3d_2/kernel/v
�
*Adam/conv3d_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_2/kernel/v*+
_output_shapes
:@�*
dtype0
�
Adam/conv3d_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/conv3d_1/bias/v
y
(Adam/conv3d_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_1/bias/v*
_output_shapes
:@*
dtype0
�
Adam/conv3d_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*'
shared_nameAdam/conv3d_1/kernel/v
�
*Adam/conv3d_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_1/kernel/v**
_output_shapes
:@@*
dtype0
|
Adam/conv3d/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*#
shared_nameAdam/conv3d/bias/v
u
&Adam/conv3d/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv3d/bias/v*
_output_shapes
:@*
dtype0
�
Adam/conv3d/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/conv3d/kernel/v
�
(Adam/conv3d/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv3d/kernel/v**
_output_shapes
:@*
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
shape:	�$*$
shared_nameAdam/dense/kernel/m
|
'Adam/dense/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/m*
_output_shapes
:	�$*
dtype0
�
Adam/conv3d_7/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*%
shared_nameAdam/conv3d_7/bias/m
z
(Adam/conv3d_7/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_7/bias/m*
_output_shapes	
:�*
dtype0
�
Adam/conv3d_7/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*!
shape:��*'
shared_nameAdam/conv3d_7/kernel/m
�
*Adam/conv3d_7/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_7/kernel/m*,
_output_shapes
:��*
dtype0
�
Adam/conv3d_6/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*%
shared_nameAdam/conv3d_6/bias/m
z
(Adam/conv3d_6/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_6/bias/m*
_output_shapes	
:�*
dtype0
�
Adam/conv3d_6/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*!
shape:��*'
shared_nameAdam/conv3d_6/kernel/m
�
*Adam/conv3d_6/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_6/kernel/m*,
_output_shapes
:��*
dtype0
�
Adam/conv3d_5/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*%
shared_nameAdam/conv3d_5/bias/m
z
(Adam/conv3d_5/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_5/bias/m*
_output_shapes	
:�*
dtype0
�
Adam/conv3d_5/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*!
shape:��*'
shared_nameAdam/conv3d_5/kernel/m
�
*Adam/conv3d_5/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_5/kernel/m*,
_output_shapes
:��*
dtype0
�
Adam/conv3d_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*%
shared_nameAdam/conv3d_4/bias/m
z
(Adam/conv3d_4/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_4/bias/m*
_output_shapes	
:�*
dtype0
�
Adam/conv3d_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*!
shape:��*'
shared_nameAdam/conv3d_4/kernel/m
�
*Adam/conv3d_4/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_4/kernel/m*,
_output_shapes
:��*
dtype0
�
Adam/conv3d_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*%
shared_nameAdam/conv3d_3/bias/m
z
(Adam/conv3d_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_3/bias/m*
_output_shapes	
:�*
dtype0
�
Adam/conv3d_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*!
shape:��*'
shared_nameAdam/conv3d_3/kernel/m
�
*Adam/conv3d_3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_3/kernel/m*,
_output_shapes
:��*
dtype0
�
Adam/conv3d_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*%
shared_nameAdam/conv3d_2/bias/m
z
(Adam/conv3d_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_2/bias/m*
_output_shapes	
:�*
dtype0
�
Adam/conv3d_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0* 
shape:@�*'
shared_nameAdam/conv3d_2/kernel/m
�
*Adam/conv3d_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_2/kernel/m*+
_output_shapes
:@�*
dtype0
�
Adam/conv3d_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/conv3d_1/bias/m
y
(Adam/conv3d_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_1/bias/m*
_output_shapes
:@*
dtype0
�
Adam/conv3d_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*'
shared_nameAdam/conv3d_1/kernel/m
�
*Adam/conv3d_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_1/kernel/m**
_output_shapes
:@@*
dtype0
|
Adam/conv3d/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*#
shared_nameAdam/conv3d/bias/m
u
&Adam/conv3d/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv3d/bias/m*
_output_shapes
:@*
dtype0
�
Adam/conv3d/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/conv3d/kernel/m
�
(Adam/conv3d/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv3d/kernel/m**
_output_shapes
:@*
dtype0
x
add_metric_1/countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameadd_metric_1/count
q
&add_metric_1/count/Read/ReadVariableOpReadVariableOpadd_metric_1/count*
_output_shapes
: *
dtype0
x
add_metric_1/totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameadd_metric_1/total
q
&add_metric_1/total/Read/ReadVariableOpReadVariableOpadd_metric_1/total*
_output_shapes
: *
dtype0
t
add_metric/countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameadd_metric/count
m
$add_metric/count/Read/ReadVariableOpReadVariableOpadd_metric/count*
_output_shapes
: *
dtype0
t
add_metric/totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameadd_metric/total
m
$add_metric/total/Read/ReadVariableOpReadVariableOpadd_metric/total*
_output_shapes
: *
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
shape:	�$*
shared_namedense/kernel
n
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes
:	�$*
dtype0
s
conv3d_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nameconv3d_7/bias
l
!conv3d_7/bias/Read/ReadVariableOpReadVariableOpconv3d_7/bias*
_output_shapes	
:�*
dtype0
�
conv3d_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*!
shape:��* 
shared_nameconv3d_7/kernel
�
#conv3d_7/kernel/Read/ReadVariableOpReadVariableOpconv3d_7/kernel*,
_output_shapes
:��*
dtype0
s
conv3d_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nameconv3d_6/bias
l
!conv3d_6/bias/Read/ReadVariableOpReadVariableOpconv3d_6/bias*
_output_shapes	
:�*
dtype0
�
conv3d_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*!
shape:��* 
shared_nameconv3d_6/kernel
�
#conv3d_6/kernel/Read/ReadVariableOpReadVariableOpconv3d_6/kernel*,
_output_shapes
:��*
dtype0
s
conv3d_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nameconv3d_5/bias
l
!conv3d_5/bias/Read/ReadVariableOpReadVariableOpconv3d_5/bias*
_output_shapes	
:�*
dtype0
�
conv3d_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*!
shape:��* 
shared_nameconv3d_5/kernel
�
#conv3d_5/kernel/Read/ReadVariableOpReadVariableOpconv3d_5/kernel*,
_output_shapes
:��*
dtype0
s
conv3d_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nameconv3d_4/bias
l
!conv3d_4/bias/Read/ReadVariableOpReadVariableOpconv3d_4/bias*
_output_shapes	
:�*
dtype0
�
conv3d_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*!
shape:��* 
shared_nameconv3d_4/kernel
�
#conv3d_4/kernel/Read/ReadVariableOpReadVariableOpconv3d_4/kernel*,
_output_shapes
:��*
dtype0
s
conv3d_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nameconv3d_3/bias
l
!conv3d_3/bias/Read/ReadVariableOpReadVariableOpconv3d_3/bias*
_output_shapes	
:�*
dtype0
�
conv3d_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*!
shape:��* 
shared_nameconv3d_3/kernel
�
#conv3d_3/kernel/Read/ReadVariableOpReadVariableOpconv3d_3/kernel*,
_output_shapes
:��*
dtype0
s
conv3d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nameconv3d_2/bias
l
!conv3d_2/bias/Read/ReadVariableOpReadVariableOpconv3d_2/bias*
_output_shapes	
:�*
dtype0
�
conv3d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0* 
shape:@�* 
shared_nameconv3d_2/kernel
�
#conv3d_2/kernel/Read/ReadVariableOpReadVariableOpconv3d_2/kernel*+
_output_shapes
:@�*
dtype0
r
conv3d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv3d_1/bias
k
!conv3d_1/bias/Read/ReadVariableOpReadVariableOpconv3d_1/bias*
_output_shapes
:@*
dtype0
�
conv3d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@* 
shared_nameconv3d_1/kernel

#conv3d_1/kernel/Read/ReadVariableOpReadVariableOpconv3d_1/kernel**
_output_shapes
:@@*
dtype0
n
conv3d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv3d/bias
g
conv3d/bias/Read/ReadVariableOpReadVariableOpconv3d/bias*
_output_shapes
:@*
dtype0
�
conv3d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv3d/kernel
{
!conv3d/kernel/Read/ReadVariableOpReadVariableOpconv3d/kernel**
_output_shapes
:@*
dtype0
J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
L
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *  �?
L
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *  �?

NoOpNoOp
��
Const_3Const"/device:CPU:0*
_output_shapes
: *
dtype0*�
valueֶBҶ Bʶ
�

layer-0
layer-1
layer-2
layer-3
layer_with_weights-0
layer-4
layer-5
layer-6
layer-7
	layer-8

layer-9
layer-10
layer-11
layer-12
layer-13
layer-14
layer-15
layer-16
layer-17
layer-18
layer-19
layer-20
layer-21
layer-22
layer-23
layer-24
layer-25
layer-26
layer-27
layer-28
layer-29
layer-30
 layer-31
!layer-32
"layer-33
#layer-34
$layer-35
%layer-36
&layer-37
'layer-38
(layer-39
)layer-40
*layer-41
+layer-42
,layer-43
-layer-44
.layer-45
/layer-46
0layer-47
1layer-48
2layer-49
3layer-50
4layer-51
5layer-52
6layer-53
7layer-54
8layer-55
9layer-56
:layer-57
;layer-58
<layer-59
=layer-60
>layer-61
?layer-62
@layer-63
Alayer-64
Blayer-65
Clayer-66
Dlayer-67
Elayer-68
Flayer-69
Glayer-70
Hlayer-71
Ilayer-72
Jlayer-73
Klayer-74
Llayer-75
M	variables
Ntrainable_variables
Oregularization_losses
P	keras_api
Q__call__
*R&call_and_return_all_conditional_losses
S_default_save_signature
T	optimizer
Uloss
V
signatures*
* 
* 

W_init_input_shape* 

X_init_input_shape* 
�
Ylayer-0
Zlayer_with_weights-0
Zlayer-1
[layer_with_weights-1
[layer-2
\layer-3
]layer_with_weights-2
]layer-4
^layer_with_weights-3
^layer-5
_layer-6
`layer_with_weights-4
`layer-7
alayer_with_weights-5
alayer-8
blayer-9
clayer_with_weights-6
clayer-10
dlayer_with_weights-7
dlayer-11
elayer-12
flayer-13
glayer_with_weights-8
glayer-14
h	variables
itrainable_variables
jregularization_losses
k	keras_api
l__call__
*m&call_and_return_all_conditional_losses*

n	keras_api* 

o	keras_api* 

p	keras_api* 

q	keras_api* 

r	keras_api* 

s	keras_api* 

t	keras_api* 

u	keras_api* 

v	keras_api* 

w	keras_api* 

x	keras_api* 

y	keras_api* 

z	keras_api* 

{	keras_api* 

|	keras_api* 

}	keras_api* 

~	keras_api* 

	keras_api* 

�	keras_api* 

�	keras_api* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 

�	keras_api* 

�	keras_api* 

�	keras_api* 

�	keras_api* 

�	keras_api* 

�	keras_api* 

�	keras_api* 

�	keras_api* 

�	keras_api* 

�	keras_api* 

�	keras_api* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses*

�	keras_api* 

�	keras_api* 

�	keras_api* 

�	keras_api* 

�	keras_api* 

�	keras_api* 

�	keras_api* 

�	keras_api* 

�	keras_api* 

�	keras_api* 

�	keras_api* 

�	keras_api* 

�	keras_api* 

�	keras_api* 

�	keras_api* 

�	keras_api* 

�	keras_api* 

�	keras_api* 

�	keras_api* 

�	keras_api* 

�	keras_api* 

�	keras_api* 

�	keras_api* 

�	keras_api* 

�	keras_api* 

�	keras_api* 

�	keras_api* 

�	keras_api* 

�	keras_api* 

�	keras_api* 

�	keras_api* 

�	keras_api* 

�	keras_api* 

�	keras_api* 

�	keras_api* 

�	keras_api* 

�	keras_api* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses*
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17*
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
M	variables
Ntrainable_variables
Oregularization_losses
Q__call__
S_default_save_signature
*R&call_and_return_all_conditional_losses
&R"call_and_return_conditional_losses*
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
�
	�iter
�beta_1
�beta_2

�decay
�learning_rate	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�*
* 

�serving_default* 
* 
* 
* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias*
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17*
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17*
J
�0
�1
�2
�3
�4
�5
�6
�7
�8* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
h	variables
itrainable_variables
jregularization_losses
l__call__
*m&call_and_return_all_conditional_losses
&m"call_and_return_conditional_losses*
:
�trace_0
�trace_1
�trace_2
�trace_3* 
:
�trace_0
�trace_1
�trace_2
�trace_3* 
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
* 
* 
* 
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
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
MG
VARIABLE_VALUEconv3d/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
KE
VARIABLE_VALUEconv3d/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEconv3d_1/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEconv3d_1/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEconv3d_2/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEconv3d_2/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEconv3d_3/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEconv3d_3/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEconv3d_4/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEconv3d_4/bias&variables/9/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEconv3d_5/kernel'variables/10/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEconv3d_5/bias'variables/11/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEconv3d_6/kernel'variables/12/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEconv3d_6/bias'variables/13/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEconv3d_7/kernel'variables/14/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEconv3d_7/bias'variables/15/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEdense/kernel'variables/16/.ATTRIBUTES/VARIABLE_VALUE*
KE
VARIABLE_VALUE
dense/bias'variables/17/.ATTRIBUTES/VARIABLE_VALUE*
* 
�
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
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
 31
!32
"33
#34
$35
%36
&37
'38
(39
)40
*41
+42
,43
-44
.45
/46
047
148
249
350
451
552
653
754
855
956
:57
;58
<59
=60
>61
?62
@63
A64
B65
C66
D67
E68
F69
G70
H71
I72
J73
K74
L75*

�0
�1
�2*
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

�0
�1*

�0
�1*


�0* 
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
* 

�0
�1*

�0
�1*


�0* 
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
�trace_0* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

�0
�1*

�0
�1*


�0* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 

�0
�1*

�0
�1*


�0* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

�0
�1*

�0
�1*


�0* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 

�0
�1*

�0
�1*


�0* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

�0
�1*

�0
�1*


�0* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 

�0
�1*

�0
�1*


�0* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

�0
�1*

�0
�1*


�0* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 

�trace_0* 

�trace_0* 

�trace_0* 

�trace_0* 

�trace_0* 

�trace_0* 

�trace_0* 

�trace_0* 

�trace_0* 
* 
r
Y0
Z1
[2
\3
]4
^5
_6
`7
a8
b9
c10
d11
e12
f13
g14*
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

�0*
* 

�	total_mse*
* 
* 
* 
* 

�0*
* 

�	corr_loss*
* 
* 
<
�	variables
�	keras_api

�total

�count*
<
�	variables
�	keras_api

�total

�count*
<
�	variables
�	keras_api

�total

�count*
* 
* 
* 


�0* 
* 
* 
* 
* 
* 
* 


�0* 
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


�0* 
* 
* 
* 
* 
* 
* 


�0* 
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


�0* 
* 
* 
* 
* 
* 
* 


�0* 
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


�0* 
* 
* 
* 
* 
* 
* 


�0* 
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


�0* 
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

�0
�1*

�	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

�0
�1*

�	variables*
^X
VARIABLE_VALUEadd_metric/total4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEadd_metric/count4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*

�0
�1*

�	variables*
`Z
VARIABLE_VALUEadd_metric_1/total4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEadd_metric_1/count4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUEAdam/conv3d/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUEAdam/conv3d/bias/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUEAdam/conv3d_1/kernel/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUEAdam/conv3d_1/bias/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUEAdam/conv3d_2/kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUEAdam/conv3d_2/bias/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUEAdam/conv3d_3/kernel/mBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUEAdam/conv3d_3/bias/mBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUEAdam/conv3d_4/kernel/mBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUEAdam/conv3d_4/bias/mBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUEAdam/conv3d_5/kernel/mCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEAdam/conv3d_5/bias/mCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUEAdam/conv3d_6/kernel/mCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEAdam/conv3d_6/bias/mCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUEAdam/conv3d_7/kernel/mCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEAdam/conv3d_7/bias/mCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUEAdam/dense/kernel/mCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUEAdam/dense/bias/mCvariables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUEAdam/conv3d/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUEAdam/conv3d/bias/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUEAdam/conv3d_1/kernel/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUEAdam/conv3d_1/bias/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUEAdam/conv3d_2/kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUEAdam/conv3d_2/bias/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUEAdam/conv3d_3/kernel/vBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUEAdam/conv3d_3/bias/vBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUEAdam/conv3d_4/kernel/vBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUEAdam/conv3d_4/bias/vBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUEAdam/conv3d_5/kernel/vCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEAdam/conv3d_5/bias/vCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUEAdam/conv3d_6/kernel/vCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEAdam/conv3d_6/bias/vCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUEAdam/conv3d_7/kernel/vCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEAdam/conv3d_7/bias/vCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUEAdam/dense/kernel/vCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUEAdam/dense/bias/vCvariables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�
serving_default_input_2Placeholder*3
_output_shapes!
:���������22*
dtype0*(
shape:���������22
�
serving_default_input_3Placeholder*3
_output_shapes!
:���������22*
dtype0*(
shape:���������22
z
serving_default_input_4Placeholder*'
_output_shapes
:���������*
dtype0*
shape:���������
z
serving_default_input_5Placeholder*'
_output_shapes
:���������*
dtype0*
shape:���������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_2serving_default_input_3serving_default_input_4serving_default_input_5conv3d/kernelconv3d/biasconv3d_1/kernelconv3d_1/biasconv3d_2/kernelconv3d_2/biasconv3d_3/kernelconv3d_3/biasconv3d_4/kernelconv3d_4/biasconv3d_5/kernelconv3d_5/biasconv3d_6/kernelconv3d_6/biasconv3d_7/kernelconv3d_7/biasdense/kernel
dense/biasadd_metric/totaladd_metric/countConstConst_1Const_2add_metric_1/totaladd_metric_1/count*(
Tin!
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:���������:���������*4
_read_only_resource_inputs
	
*4
config_proto$"

CPU

GPU(2*0J 8� *,
f'R%
#__inference_signature_wrapper_15555
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename!conv3d/kernel/Read/ReadVariableOpconv3d/bias/Read/ReadVariableOp#conv3d_1/kernel/Read/ReadVariableOp!conv3d_1/bias/Read/ReadVariableOp#conv3d_2/kernel/Read/ReadVariableOp!conv3d_2/bias/Read/ReadVariableOp#conv3d_3/kernel/Read/ReadVariableOp!conv3d_3/bias/Read/ReadVariableOp#conv3d_4/kernel/Read/ReadVariableOp!conv3d_4/bias/Read/ReadVariableOp#conv3d_5/kernel/Read/ReadVariableOp!conv3d_5/bias/Read/ReadVariableOp#conv3d_6/kernel/Read/ReadVariableOp!conv3d_6/bias/Read/ReadVariableOp#conv3d_7/kernel/Read/ReadVariableOp!conv3d_7/bias/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp$add_metric/total/Read/ReadVariableOp$add_metric/count/Read/ReadVariableOp&add_metric_1/total/Read/ReadVariableOp&add_metric_1/count/Read/ReadVariableOp(Adam/conv3d/kernel/m/Read/ReadVariableOp&Adam/conv3d/bias/m/Read/ReadVariableOp*Adam/conv3d_1/kernel/m/Read/ReadVariableOp(Adam/conv3d_1/bias/m/Read/ReadVariableOp*Adam/conv3d_2/kernel/m/Read/ReadVariableOp(Adam/conv3d_2/bias/m/Read/ReadVariableOp*Adam/conv3d_3/kernel/m/Read/ReadVariableOp(Adam/conv3d_3/bias/m/Read/ReadVariableOp*Adam/conv3d_4/kernel/m/Read/ReadVariableOp(Adam/conv3d_4/bias/m/Read/ReadVariableOp*Adam/conv3d_5/kernel/m/Read/ReadVariableOp(Adam/conv3d_5/bias/m/Read/ReadVariableOp*Adam/conv3d_6/kernel/m/Read/ReadVariableOp(Adam/conv3d_6/bias/m/Read/ReadVariableOp*Adam/conv3d_7/kernel/m/Read/ReadVariableOp(Adam/conv3d_7/bias/m/Read/ReadVariableOp'Adam/dense/kernel/m/Read/ReadVariableOp%Adam/dense/bias/m/Read/ReadVariableOp(Adam/conv3d/kernel/v/Read/ReadVariableOp&Adam/conv3d/bias/v/Read/ReadVariableOp*Adam/conv3d_1/kernel/v/Read/ReadVariableOp(Adam/conv3d_1/bias/v/Read/ReadVariableOp*Adam/conv3d_2/kernel/v/Read/ReadVariableOp(Adam/conv3d_2/bias/v/Read/ReadVariableOp*Adam/conv3d_3/kernel/v/Read/ReadVariableOp(Adam/conv3d_3/bias/v/Read/ReadVariableOp*Adam/conv3d_4/kernel/v/Read/ReadVariableOp(Adam/conv3d_4/bias/v/Read/ReadVariableOp*Adam/conv3d_5/kernel/v/Read/ReadVariableOp(Adam/conv3d_5/bias/v/Read/ReadVariableOp*Adam/conv3d_6/kernel/v/Read/ReadVariableOp(Adam/conv3d_6/bias/v/Read/ReadVariableOp*Adam/conv3d_7/kernel/v/Read/ReadVariableOp(Adam/conv3d_7/bias/v/Read/ReadVariableOp'Adam/dense/kernel/v/Read/ReadVariableOp%Adam/dense/bias/v/Read/ReadVariableOpConst_3*N
TinG
E2C	*
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
__inference__traced_save_17484
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv3d/kernelconv3d/biasconv3d_1/kernelconv3d_1/biasconv3d_2/kernelconv3d_2/biasconv3d_3/kernelconv3d_3/biasconv3d_4/kernelconv3d_4/biasconv3d_5/kernelconv3d_5/biasconv3d_6/kernelconv3d_6/biasconv3d_7/kernelconv3d_7/biasdense/kernel
dense/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcountadd_metric/totaladd_metric/countadd_metric_1/totaladd_metric_1/countAdam/conv3d/kernel/mAdam/conv3d/bias/mAdam/conv3d_1/kernel/mAdam/conv3d_1/bias/mAdam/conv3d_2/kernel/mAdam/conv3d_2/bias/mAdam/conv3d_3/kernel/mAdam/conv3d_3/bias/mAdam/conv3d_4/kernel/mAdam/conv3d_4/bias/mAdam/conv3d_5/kernel/mAdam/conv3d_5/bias/mAdam/conv3d_6/kernel/mAdam/conv3d_6/bias/mAdam/conv3d_7/kernel/mAdam/conv3d_7/bias/mAdam/dense/kernel/mAdam/dense/bias/mAdam/conv3d/kernel/vAdam/conv3d/bias/vAdam/conv3d_1/kernel/vAdam/conv3d_1/bias/vAdam/conv3d_2/kernel/vAdam/conv3d_2/bias/vAdam/conv3d_3/kernel/vAdam/conv3d_3/bias/vAdam/conv3d_4/kernel/vAdam/conv3d_4/bias/vAdam/conv3d_5/kernel/vAdam/conv3d_5/bias/vAdam/conv3d_6/kernel/vAdam/conv3d_6/bias/vAdam/conv3d_7/kernel/vAdam/conv3d_7/bias/vAdam/dense/kernel/vAdam/dense/bias/v*M
TinF
D2B*
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
!__inference__traced_restore_17689��"
�
�
C__inference_conv3d_7_layer_call_and_return_conditional_losses_13360

inputs>
conv3d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv3D/ReadVariableOp�1conv3d_7/kernel/Regularizer/Square/ReadVariableOp�
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource*,
_output_shapes
:��*
dtype0�
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :����������*
paddingSAME*
strides	
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :����������]
ReluReluBiasAdd:output:0*
T0*4
_output_shapes"
 :�����������
1conv3d_7/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv3d_readvariableop_resource*,
_output_shapes
:��*
dtype0�
"conv3d_7/kernel/Regularizer/SquareSquare9conv3d_7/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*,
_output_shapes
:��~
!conv3d_7/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
conv3d_7/kernel/Regularizer/SumSum&conv3d_7/kernel/Regularizer/Square:y:0*conv3d_7/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!conv3d_7/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    �
conv3d_7/kernel/Regularizer/mulMul*conv3d_7/kernel/Regularizer/mul/x:output:0(conv3d_7/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: n
IdentityIdentityRelu:activations:0^NoOp*
T0*4
_output_shapes"
 :�����������
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp2^conv3d_7/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp2f
1conv3d_7/kernel/Regularizer/Square/ReadVariableOp1conv3d_7/kernel/Regularizer/Square/ReadVariableOp:\ X
4
_output_shapes"
 :����������
 
_user_specified_nameinputs
�
�
__inference_loss_fn_7_17248Z
:conv3d_7_kernel_regularizer_square_readvariableop_resource:��
identity��1conv3d_7/kernel/Regularizer/Square/ReadVariableOp�
1conv3d_7/kernel/Regularizer/Square/ReadVariableOpReadVariableOp:conv3d_7_kernel_regularizer_square_readvariableop_resource*,
_output_shapes
:��*
dtype0�
"conv3d_7/kernel/Regularizer/SquareSquare9conv3d_7/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*,
_output_shapes
:��~
!conv3d_7/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
conv3d_7/kernel/Regularizer/SumSum&conv3d_7/kernel/Regularizer/Square:y:0*conv3d_7/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!conv3d_7/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    �
conv3d_7/kernel/Regularizer/mulMul*conv3d_7/kernel/Regularizer/mul/x:output:0(conv3d_7/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: a
IdentityIdentity#conv3d_7/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: z
NoOpNoOp2^conv3d_7/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2f
1conv3d_7/kernel/Regularizer/Square/ReadVariableOp1conv3d_7/kernel/Regularizer/Square/ReadVariableOp
�
�
__inference_loss_fn_1_17182X
:conv3d_1_kernel_regularizer_square_readvariableop_resource:@@
identity��1conv3d_1/kernel/Regularizer/Square/ReadVariableOp�
1conv3d_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp:conv3d_1_kernel_regularizer_square_readvariableop_resource**
_output_shapes
:@@*
dtype0�
"conv3d_1/kernel/Regularizer/SquareSquare9conv3d_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0**
_output_shapes
:@@~
!conv3d_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
conv3d_1/kernel/Regularizer/SumSum&conv3d_1/kernel/Regularizer/Square:y:0*conv3d_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!conv3d_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    �
conv3d_1/kernel/Regularizer/mulMul*conv3d_1/kernel/Regularizer/mul/x:output:0(conv3d_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: a
IdentityIdentity#conv3d_1/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: z
NoOpNoOp2^conv3d_1/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2f
1conv3d_1/kernel/Regularizer/Square/ReadVariableOp1conv3d_1/kernel/Regularizer/Square/ReadVariableOp
�
�
C__inference_conv3d_7_layer_call_and_return_conditional_losses_17114

inputs>
conv3d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv3D/ReadVariableOp�1conv3d_7/kernel/Regularizer/Square/ReadVariableOp�
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource*,
_output_shapes
:��*
dtype0�
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :����������*
paddingSAME*
strides	
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :����������]
ReluReluBiasAdd:output:0*
T0*4
_output_shapes"
 :�����������
1conv3d_7/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv3d_readvariableop_resource*,
_output_shapes
:��*
dtype0�
"conv3d_7/kernel/Regularizer/SquareSquare9conv3d_7/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*,
_output_shapes
:��~
!conv3d_7/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
conv3d_7/kernel/Regularizer/SumSum&conv3d_7/kernel/Regularizer/Square:y:0*conv3d_7/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!conv3d_7/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    �
conv3d_7/kernel/Regularizer/mulMul*conv3d_7/kernel/Regularizer/mul/x:output:0(conv3d_7/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: n
IdentityIdentityRelu:activations:0^NoOp*
T0*4
_output_shapes"
 :�����������
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp2^conv3d_7/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp2f
1conv3d_7/kernel/Regularizer/Square/ReadVariableOp1conv3d_7/kernel/Regularizer/Square/ReadVariableOp:\ X
4
_output_shapes"
 :����������
 
_user_specified_nameinputs
�
�
C__inference_conv3d_2_layer_call_and_return_conditional_losses_16964

inputs=
conv3d_readvariableop_resource:@�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv3D/ReadVariableOp�1conv3d_2/kernel/Regularizer/Square/ReadVariableOp�
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource*+
_output_shapes
:@�*
dtype0�
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :���������
�*
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
 :���������
�]
ReluReluBiasAdd:output:0*
T0*4
_output_shapes"
 :���������
��
1conv3d_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv3d_readvariableop_resource*+
_output_shapes
:@�*
dtype0�
"conv3d_2/kernel/Regularizer/SquareSquare9conv3d_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*+
_output_shapes
:@�~
!conv3d_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
conv3d_2/kernel/Regularizer/SumSum&conv3d_2/kernel/Regularizer/Square:y:0*conv3d_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!conv3d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    �
conv3d_2/kernel/Regularizer/mulMul*conv3d_2/kernel/Regularizer/mul/x:output:0(conv3d_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: n
IdentityIdentityRelu:activations:0^NoOp*
T0*4
_output_shapes"
 :���������
��
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp2^conv3d_2/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������
@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp2f
1conv3d_2/kernel/Regularizer/Square/ReadVariableOp1conv3d_2/kernel/Regularizer/Square/ReadVariableOp:[ W
3
_output_shapes!
:���������
@
 
_user_specified_nameinputs
��
�
B__inference_model_1_layer_call_and_return_conditional_losses_15433
input_2
input_3
input_4
input_5)
model_15168:@
model_15170:@)
model_15172:@@
model_15174:@*
model_15176:@�
model_15178:	�+
model_15180:��
model_15182:	�+
model_15184:��
model_15186:	�+
model_15188:��
model_15190:	�+
model_15192:��
model_15194:	�+
model_15196:��
model_15198:	�
model_15200:	�$
model_15202:
add_metric_15339: 
add_metric_15341: 
unknown
	unknown_0
	unknown_1
add_metric_1_15371: 
add_metric_1_15373: 
identity

identity_1

identity_2��"add_metric/StatefulPartitionedCall�$add_metric_1/StatefulPartitionedCall�/conv3d/kernel/Regularizer/Square/ReadVariableOp�1conv3d_1/kernel/Regularizer/Square/ReadVariableOp�1conv3d_2/kernel/Regularizer/Square/ReadVariableOp�1conv3d_3/kernel/Regularizer/Square/ReadVariableOp�1conv3d_4/kernel/Regularizer/Square/ReadVariableOp�1conv3d_5/kernel/Regularizer/Square/ReadVariableOp�1conv3d_6/kernel/Regularizer/Square/ReadVariableOp�1conv3d_7/kernel/Regularizer/Square/ReadVariableOp�.dense/kernel/Regularizer/Square/ReadVariableOp�model/StatefulPartitionedCall�model/StatefulPartitionedCall_1�
model/StatefulPartitionedCallStatefulPartitionedCallinput_3model_15168model_15170model_15172model_15174model_15176model_15178model_15180model_15182model_15184model_15186model_15188model_15190model_15192model_15194model_15196model_15198model_15200model_15202*
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
@__inference_model_layer_call_and_return_conditional_losses_13738�
model/StatefulPartitionedCall_1StatefulPartitionedCallinput_2model_15168model_15170model_15172model_15174model_15176model_15178model_15180model_15182model_15184model_15186model_15188model_15190model_15192model_15194model_15196model_15198model_15200model_15202*
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
@__inference_model_layer_call_and_return_conditional_losses_13738U
tf.compat.v1.squeeze_1/SqueezeSqueezeinput_5*
T0*
_output_shapes
:t
tf.compat.v1.squeeze_3/SqueezeSqueeze&model/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:S
tf.compat.v1.squeeze/SqueezeSqueezeinput_4*
T0*
_output_shapes
:v
tf.compat.v1.squeeze_2/SqueezeSqueeze(model/StatefulPartitionedCall_1:output:0*
T0*
_output_shapes
:v
tf.compat.v1.squeeze_4/SqueezeSqueeze(model/StatefulPartitionedCall_1:output:0*
T0*
_output_shapes
:t
tf.compat.v1.squeeze_5/SqueezeSqueeze&model/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:U
tf.compat.v1.squeeze_7/SqueezeSqueezeinput_5*
T0*
_output_shapes
:t
tf.compat.v1.squeeze_9/SqueezeSqueeze&model/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:U
tf.compat.v1.squeeze_6/SqueezeSqueezeinput_4*
T0*
_output_shapes
:v
tf.compat.v1.squeeze_8/SqueezeSqueeze(model/StatefulPartitionedCall_1:output:0*
T0*
_output_shapes
:u
tf.compat.v1.squeeze_13/SqueezeSqueeze&model/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:V
tf.compat.v1.squeeze_12/SqueezeSqueezeinput_5*
T0*
_output_shapes
:w
tf.compat.v1.squeeze_11/SqueezeSqueeze(model/StatefulPartitionedCall_1:output:0*
T0*
_output_shapes
:V
tf.compat.v1.squeeze_10/SqueezeSqueezeinput_4*
T0*
_output_shapes
:�
tf.math.subtract_1/SubSub'tf.compat.v1.squeeze_1/Squeeze:output:0'tf.compat.v1.squeeze_3/Squeeze:output:0*
T0*
_output_shapes
:�
tf.math.subtract/SubSub%tf.compat.v1.squeeze/Squeeze:output:0'tf.compat.v1.squeeze_2/Squeeze:output:0*
T0*
_output_shapes
:�
tf.math.subtract_2/SubSub'tf.compat.v1.squeeze_4/Squeeze:output:0'tf.compat.v1.squeeze_5/Squeeze:output:0*
T0*
_output_shapes
:�
tf.math.subtract_5/SubSub'tf.compat.v1.squeeze_7/Squeeze:output:0'tf.compat.v1.squeeze_9/Squeeze:output:0*
T0*
_output_shapes
:�
tf.math.subtract_4/SubSub'tf.compat.v1.squeeze_6/Squeeze:output:0'tf.compat.v1.squeeze_8/Squeeze:output:0*
T0*
_output_shapes
:m
tf.math.reduce_mean_8/RankRank(tf.compat.v1.squeeze_13/Squeeze:output:0*
T0*
_output_shapes
: c
!tf.math.reduce_mean_8/range/startConst*
_output_shapes
: *
dtype0*
value	B : c
!tf.math.reduce_mean_8/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
tf.math.reduce_mean_8/rangeRange*tf.math.reduce_mean_8/range/start:output:0#tf.math.reduce_mean_8/Rank:output:0*tf.math.reduce_mean_8/range/delta:output:0*#
_output_shapes
:����������
tf.math.reduce_mean_8/MeanMean(tf.compat.v1.squeeze_13/Squeeze:output:0$tf.math.reduce_mean_8/range:output:0*
T0*
_output_shapes
: m
tf.math.reduce_mean_7/RankRank(tf.compat.v1.squeeze_12/Squeeze:output:0*
T0*
_output_shapes
: c
!tf.math.reduce_mean_7/range/startConst*
_output_shapes
: *
dtype0*
value	B : c
!tf.math.reduce_mean_7/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
tf.math.reduce_mean_7/rangeRange*tf.math.reduce_mean_7/range/start:output:0#tf.math.reduce_mean_7/Rank:output:0*tf.math.reduce_mean_7/range/delta:output:0*#
_output_shapes
:����������
tf.math.reduce_mean_7/MeanMean(tf.compat.v1.squeeze_12/Squeeze:output:0$tf.math.reduce_mean_7/range:output:0*
T0*
_output_shapes
: m
tf.math.reduce_mean_6/RankRank(tf.compat.v1.squeeze_11/Squeeze:output:0*
T0*
_output_shapes
: c
!tf.math.reduce_mean_6/range/startConst*
_output_shapes
: *
dtype0*
value	B : c
!tf.math.reduce_mean_6/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
tf.math.reduce_mean_6/rangeRange*tf.math.reduce_mean_6/range/start:output:0#tf.math.reduce_mean_6/Rank:output:0*tf.math.reduce_mean_6/range/delta:output:0*#
_output_shapes
:����������
tf.math.reduce_mean_6/MeanMean(tf.compat.v1.squeeze_11/Squeeze:output:0$tf.math.reduce_mean_6/range:output:0*
T0*
_output_shapes
: m
tf.math.reduce_mean_5/RankRank(tf.compat.v1.squeeze_10/Squeeze:output:0*
T0*
_output_shapes
: c
!tf.math.reduce_mean_5/range/startConst*
_output_shapes
: *
dtype0*
value	B : c
!tf.math.reduce_mean_5/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
tf.math.reduce_mean_5/rangeRange*tf.math.reduce_mean_5/range/start:output:0#tf.math.reduce_mean_5/Rank:output:0*tf.math.reduce_mean_5/range/delta:output:0*#
_output_shapes
:����������
tf.math.reduce_mean_5/MeanMean(tf.compat.v1.squeeze_10/Squeeze:output:0$tf.math.reduce_mean_5/range:output:0*
T0*
_output_shapes
: `
tf.math.square_1/SquareSquaretf.math.subtract_1/Sub:z:0*
T0*
_output_shapes
:\
tf.math.square/SquareSquaretf.math.subtract/Sub:z:0*
T0*
_output_shapes
:`
tf.math.square_2/SquareSquaretf.math.subtract_2/Sub:z:0*
T0*
_output_shapes
:`
tf.math.square_4/SquareSquaretf.math.subtract_5/Sub:z:0*
T0*
_output_shapes
:`
tf.math.square_3/SquareSquaretf.math.subtract_4/Sub:z:0*
T0*
_output_shapes
:�
tf.math.subtract_10/SubSub(tf.compat.v1.squeeze_13/Squeeze:output:0#tf.math.reduce_mean_8/Mean:output:0*
T0*
_output_shapes
:�
tf.math.subtract_9/SubSub(tf.compat.v1.squeeze_12/Squeeze:output:0#tf.math.reduce_mean_7/Mean:output:0*
T0*
_output_shapes
:�
tf.math.subtract_7/SubSub(tf.compat.v1.squeeze_11/Squeeze:output:0#tf.math.reduce_mean_6/Mean:output:0*
T0*
_output_shapes
:�
tf.math.subtract_6/SubSub(tf.compat.v1.squeeze_10/Squeeze:output:0#tf.math.reduce_mean_5/Mean:output:0*
T0*
_output_shapes
:\
tf.math.reduce_mean/RankRanktf.math.square/Square:y:0*
T0*
_output_shapes
: a
tf.math.reduce_mean/range/startConst*
_output_shapes
: *
dtype0*
value	B : a
tf.math.reduce_mean/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
tf.math.reduce_mean/rangeRange(tf.math.reduce_mean/range/start:output:0!tf.math.reduce_mean/Rank:output:0(tf.math.reduce_mean/range/delta:output:0*#
_output_shapes
:����������
tf.math.reduce_mean/MeanMeantf.math.square/Square:y:0"tf.math.reduce_mean/range:output:0*
T0*
_output_shapes
: `
tf.math.reduce_mean_1/RankRanktf.math.square_1/Square:y:0*
T0*
_output_shapes
: c
!tf.math.reduce_mean_1/range/startConst*
_output_shapes
: *
dtype0*
value	B : c
!tf.math.reduce_mean_1/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
tf.math.reduce_mean_1/rangeRange*tf.math.reduce_mean_1/range/start:output:0#tf.math.reduce_mean_1/Rank:output:0*tf.math.reduce_mean_1/range/delta:output:0*#
_output_shapes
:����������
tf.math.reduce_mean_1/MeanMeantf.math.square_1/Square:y:0$tf.math.reduce_mean_1/range:output:0*
T0*
_output_shapes
: ^
tf.math.maximum/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *��:�
tf.math.maximum/MaximumMaximumtf.math.square_2/Square:y:0"tf.math.maximum/Maximum/y:output:0*
T0*
_output_shapes
:`
tf.math.reduce_mean_3/RankRanktf.math.square_3/Square:y:0*
T0*
_output_shapes
: c
!tf.math.reduce_mean_3/range/startConst*
_output_shapes
: *
dtype0*
value	B : c
!tf.math.reduce_mean_3/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
tf.math.reduce_mean_3/rangeRange*tf.math.reduce_mean_3/range/start:output:0#tf.math.reduce_mean_3/Rank:output:0*tf.math.reduce_mean_3/range/delta:output:0*#
_output_shapes
:����������
tf.math.reduce_mean_3/MeanMeantf.math.square_3/Square:y:0$tf.math.reduce_mean_3/range:output:0*
T0*
_output_shapes
: `
tf.math.reduce_mean_4/RankRanktf.math.square_4/Square:y:0*
T0*
_output_shapes
: c
!tf.math.reduce_mean_4/range/startConst*
_output_shapes
: *
dtype0*
value	B : c
!tf.math.reduce_mean_4/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
tf.math.reduce_mean_4/rangeRange*tf.math.reduce_mean_4/range/start:output:0#tf.math.reduce_mean_4/Rank:output:0*tf.math.reduce_mean_4/range/delta:output:0*#
_output_shapes
:����������
tf.math.reduce_mean_4/MeanMeantf.math.square_4/Square:y:0$tf.math.reduce_mean_4/range:output:0*
T0*
_output_shapes
: y
tf.math.multiply_3/MulMultf.math.subtract_9/Sub:z:0tf.math.subtract_10/Sub:z:0*
T0*
_output_shapes
:x
tf.math.multiply_1/MulMultf.math.subtract_6/Sub:z:0tf.math.subtract_7/Sub:z:0*
T0*
_output_shapes
:a
tf.math.square_8/SquareSquaretf.math.subtract_10/Sub:z:0*
T0*
_output_shapes
:`
tf.math.square_7/SquareSquaretf.math.subtract_9/Sub:z:0*
T0*
_output_shapes
:`
tf.math.square_6/SquareSquaretf.math.subtract_7/Sub:z:0*
T0*
_output_shapes
:`
tf.math.square_5/SquareSquaretf.math.subtract_6/Sub:z:0*
T0*
_output_shapes
:�
tf.__operators__.add/AddV2AddV2!tf.math.reduce_mean/Mean:output:0#tf.math.reduce_mean_1/Mean:output:0*
T0*
_output_shapes
: ]
tf.math.subtract_3/Sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *��:�
tf.math.subtract_3/SubSubtf.math.maximum/Maximum:z:0!tf.math.subtract_3/Sub/y:output:0*
T0*
_output_shapes
:�
tf.__operators__.add_2/AddV2AddV2#tf.math.reduce_mean_3/Mean:output:0#tf.math.reduce_mean_4/Mean:output:0*
T0*
_output_shapes
: ^
tf.math.reduce_sum_3/RankRanktf.math.multiply_3/Mul:z:0*
T0*
_output_shapes
: b
 tf.math.reduce_sum_3/range/startConst*
_output_shapes
: *
dtype0*
value	B : b
 tf.math.reduce_sum_3/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
tf.math.reduce_sum_3/rangeRange)tf.math.reduce_sum_3/range/start:output:0"tf.math.reduce_sum_3/Rank:output:0)tf.math.reduce_sum_3/range/delta:output:0*#
_output_shapes
:����������
tf.math.reduce_sum_3/SumSumtf.math.multiply_3/Mul:z:0#tf.math.reduce_sum_3/range:output:0*
T0*
_output_shapes
: \
tf.math.reduce_sum/RankRanktf.math.multiply_1/Mul:z:0*
T0*
_output_shapes
: `
tf.math.reduce_sum/range/startConst*
_output_shapes
: *
dtype0*
value	B : `
tf.math.reduce_sum/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
tf.math.reduce_sum/rangeRange'tf.math.reduce_sum/range/start:output:0 tf.math.reduce_sum/Rank:output:0'tf.math.reduce_sum/range/delta:output:0*#
_output_shapes
:���������}
tf.math.reduce_sum/SumSumtf.math.multiply_1/Mul:z:0!tf.math.reduce_sum/range:output:0*
T0*
_output_shapes
: _
tf.math.reduce_sum_4/RankRanktf.math.square_7/Square:y:0*
T0*
_output_shapes
: b
 tf.math.reduce_sum_4/range/startConst*
_output_shapes
: *
dtype0*
value	B : b
 tf.math.reduce_sum_4/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
tf.math.reduce_sum_4/rangeRange)tf.math.reduce_sum_4/range/start:output:0"tf.math.reduce_sum_4/Rank:output:0)tf.math.reduce_sum_4/range/delta:output:0*#
_output_shapes
:����������
tf.math.reduce_sum_4/SumSumtf.math.square_7/Square:y:0#tf.math.reduce_sum_4/range:output:0*
T0*
_output_shapes
: _
tf.math.reduce_sum_5/RankRanktf.math.square_8/Square:y:0*
T0*
_output_shapes
: b
 tf.math.reduce_sum_5/range/startConst*
_output_shapes
: *
dtype0*
value	B : b
 tf.math.reduce_sum_5/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
tf.math.reduce_sum_5/rangeRange)tf.math.reduce_sum_5/range/start:output:0"tf.math.reduce_sum_5/Rank:output:0)tf.math.reduce_sum_5/range/delta:output:0*#
_output_shapes
:����������
tf.math.reduce_sum_5/SumSumtf.math.square_8/Square:y:0#tf.math.reduce_sum_5/range:output:0*
T0*
_output_shapes
: _
tf.math.reduce_sum_1/RankRanktf.math.square_5/Square:y:0*
T0*
_output_shapes
: b
 tf.math.reduce_sum_1/range/startConst*
_output_shapes
: *
dtype0*
value	B : b
 tf.math.reduce_sum_1/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
tf.math.reduce_sum_1/rangeRange)tf.math.reduce_sum_1/range/start:output:0"tf.math.reduce_sum_1/Rank:output:0)tf.math.reduce_sum_1/range/delta:output:0*#
_output_shapes
:����������
tf.math.reduce_sum_1/SumSumtf.math.square_5/Square:y:0#tf.math.reduce_sum_1/range:output:0*
T0*
_output_shapes
: _
tf.math.reduce_sum_2/RankRanktf.math.square_6/Square:y:0*
T0*
_output_shapes
: b
 tf.math.reduce_sum_2/range/startConst*
_output_shapes
: *
dtype0*
value	B : b
 tf.math.reduce_sum_2/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
tf.math.reduce_sum_2/rangeRange)tf.math.reduce_sum_2/range/start:output:0"tf.math.reduce_sum_2/Rank:output:0)tf.math.reduce_sum_2/range/delta:output:0*#
_output_shapes
:����������
tf.math.reduce_sum_2/SumSumtf.math.square_6/Square:y:0#tf.math.reduce_sum_2/range:output:0*
T0*
_output_shapes
: _
tf.math.reduce_mean_2/RankRanktf.math.subtract_3/Sub:z:0*
T0*
_output_shapes
: c
!tf.math.reduce_mean_2/range/startConst*
_output_shapes
: *
dtype0*
value	B : c
!tf.math.reduce_mean_2/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
tf.math.reduce_mean_2/rangeRange*tf.math.reduce_mean_2/range/start:output:0#tf.math.reduce_mean_2/Rank:output:0*tf.math.reduce_mean_2/range/delta:output:0*#
_output_shapes
:����������
tf.math.reduce_mean_2/MeanMeantf.math.subtract_3/Sub:z:0$tf.math.reduce_mean_2/range:output:0*
T0*
_output_shapes
: �
"add_metric/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_2/AddV2:z:0add_metric_15339add_metric_15341*
Tin
2*
Tout
2*
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
GPU(2*0J 8� *N
fIRG
E__inference_add_metric_layer_call_and_return_conditional_losses_14235�
tf.math.multiply_4/MulMul!tf.math.reduce_sum_4/Sum:output:0!tf.math.reduce_sum_5/Sum:output:0*
T0*
_output_shapes
: �
tf.math.multiply_2/MulMul!tf.math.reduce_sum_1/Sum:output:0!tf.math.reduce_sum_2/Sum:output:0*
T0*
_output_shapes
: j
tf.math.multiply/MulMulunknown#tf.math.reduce_mean_2/Mean:output:0*
T0*
_output_shapes
: X
tf.math.sqrt_1/SqrtSqrttf.math.multiply_4/Mul:z:0*
T0*
_output_shapes
: V
tf.math.sqrt/SqrtSqrttf.math.multiply_2/Mul:z:0*
T0*
_output_shapes
: �
tf.__operators__.add_1/AddV2AddV2tf.__operators__.add/AddV2:z:0tf.math.multiply/Mul:z:0*
T0*
_output_shapes
: ]
tf.__operators__.add_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *��'7�
tf.__operators__.add_4/AddV2AddV2tf.math.sqrt_1/Sqrt:y:0!tf.__operators__.add_4/y:output:0*
T0*
_output_shapes
: ]
tf.__operators__.add_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *��'7�
tf.__operators__.add_3/AddV2AddV2tf.math.sqrt/Sqrt:y:0!tf.__operators__.add_3/y:output:0*
T0*
_output_shapes
: �
add_loss/PartitionedCallPartitionedCall tf.__operators__.add_1/AddV2:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: : * 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8� *L
fGRE
C__inference_add_loss_layer_call_and_return_conditional_losses_14258�
tf.math.truediv_1/truedivRealDiv!tf.math.reduce_sum_3/Sum:output:0 tf.__operators__.add_4/AddV2:z:0*
T0*
_output_shapes
: �
tf.math.truediv/truedivRealDivtf.math.reduce_sum/Sum:output:0 tf.__operators__.add_3/AddV2:z:0*
T0*
_output_shapes
: i
tf.math.subtract_11/SubSub	unknown_0tf.math.truediv_1/truediv:z:0*
T0*
_output_shapes
: f
tf.math.subtract_8/SubSub	unknown_1tf.math.truediv/truediv:z:0*
T0*
_output_shapes
: V
tf.math.pow/Pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?o
tf.math.pow/PowPowtf.math.subtract_8/Sub:z:0tf.math.pow/Pow/y:output:0*
T0*
_output_shapes
: X
tf.math.pow_1/Pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?t
tf.math.pow_1/PowPowtf.math.subtract_11/Sub:z:0tf.math.pow_1/Pow/y:output:0*
T0*
_output_shapes
: r
tf.__operators__.add_5/AddV2AddV2tf.math.pow/Pow:z:0tf.math.pow_1/Pow:z:0*
T0*
_output_shapes
: �
$add_metric_1/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_5/AddV2:z:0add_metric_1_15371add_metric_1_15373*
Tin
2*
Tout
2*
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
GPU(2*0J 8� *P
fKRI
G__inference_add_metric_1_layer_call_and_return_conditional_losses_14291�
/conv3d/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmodel_15168**
_output_shapes
:@*
dtype0�
 conv3d/kernel/Regularizer/SquareSquare7conv3d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0**
_output_shapes
:@|
conv3d/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
conv3d/kernel/Regularizer/SumSum$conv3d/kernel/Regularizer/Square:y:0(conv3d/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: d
conv3d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    �
conv3d/kernel/Regularizer/mulMul(conv3d/kernel/Regularizer/mul/x:output:0&conv3d/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1conv3d_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmodel_15172**
_output_shapes
:@@*
dtype0�
"conv3d_1/kernel/Regularizer/SquareSquare9conv3d_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0**
_output_shapes
:@@~
!conv3d_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
conv3d_1/kernel/Regularizer/SumSum&conv3d_1/kernel/Regularizer/Square:y:0*conv3d_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!conv3d_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    �
conv3d_1/kernel/Regularizer/mulMul*conv3d_1/kernel/Regularizer/mul/x:output:0(conv3d_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1conv3d_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmodel_15176*+
_output_shapes
:@�*
dtype0�
"conv3d_2/kernel/Regularizer/SquareSquare9conv3d_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*+
_output_shapes
:@�~
!conv3d_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
conv3d_2/kernel/Regularizer/SumSum&conv3d_2/kernel/Regularizer/Square:y:0*conv3d_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!conv3d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    �
conv3d_2/kernel/Regularizer/mulMul*conv3d_2/kernel/Regularizer/mul/x:output:0(conv3d_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1conv3d_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmodel_15180*,
_output_shapes
:��*
dtype0�
"conv3d_3/kernel/Regularizer/SquareSquare9conv3d_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*,
_output_shapes
:��~
!conv3d_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
conv3d_3/kernel/Regularizer/SumSum&conv3d_3/kernel/Regularizer/Square:y:0*conv3d_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!conv3d_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    �
conv3d_3/kernel/Regularizer/mulMul*conv3d_3/kernel/Regularizer/mul/x:output:0(conv3d_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1conv3d_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmodel_15184*,
_output_shapes
:��*
dtype0�
"conv3d_4/kernel/Regularizer/SquareSquare9conv3d_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*,
_output_shapes
:��~
!conv3d_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
conv3d_4/kernel/Regularizer/SumSum&conv3d_4/kernel/Regularizer/Square:y:0*conv3d_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!conv3d_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    �
conv3d_4/kernel/Regularizer/mulMul*conv3d_4/kernel/Regularizer/mul/x:output:0(conv3d_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1conv3d_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmodel_15188*,
_output_shapes
:��*
dtype0�
"conv3d_5/kernel/Regularizer/SquareSquare9conv3d_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*,
_output_shapes
:��~
!conv3d_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
conv3d_5/kernel/Regularizer/SumSum&conv3d_5/kernel/Regularizer/Square:y:0*conv3d_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!conv3d_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    �
conv3d_5/kernel/Regularizer/mulMul*conv3d_5/kernel/Regularizer/mul/x:output:0(conv3d_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1conv3d_6/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmodel_15192*,
_output_shapes
:��*
dtype0�
"conv3d_6/kernel/Regularizer/SquareSquare9conv3d_6/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*,
_output_shapes
:��~
!conv3d_6/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
conv3d_6/kernel/Regularizer/SumSum&conv3d_6/kernel/Regularizer/Square:y:0*conv3d_6/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!conv3d_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    �
conv3d_6/kernel/Regularizer/mulMul*conv3d_6/kernel/Regularizer/mul/x:output:0(conv3d_6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1conv3d_7/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmodel_15196*,
_output_shapes
:��*
dtype0�
"conv3d_7/kernel/Regularizer/SquareSquare9conv3d_7/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*,
_output_shapes
:��~
!conv3d_7/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
conv3d_7/kernel/Regularizer/SumSum&conv3d_7/kernel/Regularizer/Square:y:0*conv3d_7/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!conv3d_7/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    �
conv3d_7/kernel/Regularizer/mulMul*conv3d_7/kernel/Regularizer/mul/x:output:0(conv3d_7/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: {
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmodel_15200*
_output_shapes
:	�$*
dtype0�
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	�$o
dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
dense/kernel/Regularizer/SumSum#dense/kernel/Regularizer/Square:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: c
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    �
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: w
IdentityIdentity(model/StatefulPartitionedCall_1:output:0^NoOp*
T0*'
_output_shapes
:���������w

Identity_1Identity&model/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������a

Identity_2Identity!add_loss/PartitionedCall:output:1^NoOp*
T0*
_output_shapes
: �
NoOpNoOp#^add_metric/StatefulPartitionedCall%^add_metric_1/StatefulPartitionedCall0^conv3d/kernel/Regularizer/Square/ReadVariableOp2^conv3d_1/kernel/Regularizer/Square/ReadVariableOp2^conv3d_2/kernel/Regularizer/Square/ReadVariableOp2^conv3d_3/kernel/Regularizer/Square/ReadVariableOp2^conv3d_4/kernel/Regularizer/Square/ReadVariableOp2^conv3d_5/kernel/Regularizer/Square/ReadVariableOp2^conv3d_6/kernel/Regularizer/Square/ReadVariableOp2^conv3d_7/kernel/Regularizer/Square/ReadVariableOp/^dense/kernel/Regularizer/Square/ReadVariableOp^model/StatefulPartitionedCall ^model/StatefulPartitionedCall_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������22:���������22:���������:���������: : : : : : : : : : : : : : : : : : : : : : : : : 2H
"add_metric/StatefulPartitionedCall"add_metric/StatefulPartitionedCall2L
$add_metric_1/StatefulPartitionedCall$add_metric_1/StatefulPartitionedCall2b
/conv3d/kernel/Regularizer/Square/ReadVariableOp/conv3d/kernel/Regularizer/Square/ReadVariableOp2f
1conv3d_1/kernel/Regularizer/Square/ReadVariableOp1conv3d_1/kernel/Regularizer/Square/ReadVariableOp2f
1conv3d_2/kernel/Regularizer/Square/ReadVariableOp1conv3d_2/kernel/Regularizer/Square/ReadVariableOp2f
1conv3d_3/kernel/Regularizer/Square/ReadVariableOp1conv3d_3/kernel/Regularizer/Square/ReadVariableOp2f
1conv3d_4/kernel/Regularizer/Square/ReadVariableOp1conv3d_4/kernel/Regularizer/Square/ReadVariableOp2f
1conv3d_5/kernel/Regularizer/Square/ReadVariableOp1conv3d_5/kernel/Regularizer/Square/ReadVariableOp2f
1conv3d_6/kernel/Regularizer/Square/ReadVariableOp1conv3d_6/kernel/Regularizer/Square/ReadVariableOp2f
1conv3d_7/kernel/Regularizer/Square/ReadVariableOp1conv3d_7/kernel/Regularizer/Square/ReadVariableOp2`
.dense/kernel/Regularizer/Square/ReadVariableOp.dense/kernel/Regularizer/Square/ReadVariableOp2>
model/StatefulPartitionedCallmodel/StatefulPartitionedCall2B
model/StatefulPartitionedCall_1model/StatefulPartitionedCall_1:\ X
3
_output_shapes!
:���������22
!
_user_specified_name	input_2:\X
3
_output_shapes!
:���������22
!
_user_specified_name	input_3:PL
'
_output_shapes
:���������
!
_user_specified_name	input_4:PL
'
_output_shapes
:���������
!
_user_specified_name	input_5:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
�
__inference_loss_fn_4_17215Z
:conv3d_4_kernel_regularizer_square_readvariableop_resource:��
identity��1conv3d_4/kernel/Regularizer/Square/ReadVariableOp�
1conv3d_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOp:conv3d_4_kernel_regularizer_square_readvariableop_resource*,
_output_shapes
:��*
dtype0�
"conv3d_4/kernel/Regularizer/SquareSquare9conv3d_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*,
_output_shapes
:��~
!conv3d_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
conv3d_4/kernel/Regularizer/SumSum&conv3d_4/kernel/Regularizer/Square:y:0*conv3d_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!conv3d_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    �
conv3d_4/kernel/Regularizer/mulMul*conv3d_4/kernel/Regularizer/mul/x:output:0(conv3d_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: a
IdentityIdentity#conv3d_4/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: z
NoOpNoOp2^conv3d_4/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2f
1conv3d_4/kernel/Regularizer/Square/ReadVariableOp1conv3d_4/kernel/Regularizer/Square/ReadVariableOp
�
�
C__inference_conv3d_3_layer_call_and_return_conditional_losses_13266

inputs>
conv3d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv3D/ReadVariableOp�1conv3d_3/kernel/Regularizer/Square/ReadVariableOp�
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource*,
_output_shapes
:��*
dtype0�
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :���������
�*
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
 :���������
�]
ReluReluBiasAdd:output:0*
T0*4
_output_shapes"
 :���������
��
1conv3d_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv3d_readvariableop_resource*,
_output_shapes
:��*
dtype0�
"conv3d_3/kernel/Regularizer/SquareSquare9conv3d_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*,
_output_shapes
:��~
!conv3d_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
conv3d_3/kernel/Regularizer/SumSum&conv3d_3/kernel/Regularizer/Square:y:0*conv3d_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!conv3d_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    �
conv3d_3/kernel/Regularizer/mulMul*conv3d_3/kernel/Regularizer/mul/x:output:0(conv3d_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: n
IdentityIdentityRelu:activations:0^NoOp*
T0*4
_output_shapes"
 :���������
��
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp2^conv3d_3/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:���������
�: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp2f
1conv3d_3/kernel/Regularizer/Square/ReadVariableOp1conv3d_3/kernel/Regularizer/Square/ReadVariableOp:\ X
4
_output_shapes"
 :���������
�
 
_user_specified_nameinputs
�
�
__inference_loss_fn_3_17204Z
:conv3d_3_kernel_regularizer_square_readvariableop_resource:��
identity��1conv3d_3/kernel/Regularizer/Square/ReadVariableOp�
1conv3d_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOp:conv3d_3_kernel_regularizer_square_readvariableop_resource*,
_output_shapes
:��*
dtype0�
"conv3d_3/kernel/Regularizer/SquareSquare9conv3d_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*,
_output_shapes
:��~
!conv3d_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
conv3d_3/kernel/Regularizer/SumSum&conv3d_3/kernel/Regularizer/Square:y:0*conv3d_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!conv3d_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    �
conv3d_3/kernel/Regularizer/mulMul*conv3d_3/kernel/Regularizer/mul/x:output:0(conv3d_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: a
IdentityIdentity#conv3d_3/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: z
NoOpNoOp2^conv3d_3/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2f
1conv3d_3/kernel/Regularizer/Square/ReadVariableOp1conv3d_3/kernel/Regularizer/Square/ReadVariableOp
��
�
B__inference_model_1_layer_call_and_return_conditional_losses_14774

inputs
inputs_1
inputs_2
inputs_3)
model_14509:@
model_14511:@)
model_14513:@@
model_14515:@*
model_14517:@�
model_14519:	�+
model_14521:��
model_14523:	�+
model_14525:��
model_14527:	�+
model_14529:��
model_14531:	�+
model_14533:��
model_14535:	�+
model_14537:��
model_14539:	�
model_14541:	�$
model_14543:
add_metric_14680: 
add_metric_14682: 
unknown
	unknown_0
	unknown_1
add_metric_1_14712: 
add_metric_1_14714: 
identity

identity_1

identity_2��"add_metric/StatefulPartitionedCall�$add_metric_1/StatefulPartitionedCall�/conv3d/kernel/Regularizer/Square/ReadVariableOp�1conv3d_1/kernel/Regularizer/Square/ReadVariableOp�1conv3d_2/kernel/Regularizer/Square/ReadVariableOp�1conv3d_3/kernel/Regularizer/Square/ReadVariableOp�1conv3d_4/kernel/Regularizer/Square/ReadVariableOp�1conv3d_5/kernel/Regularizer/Square/ReadVariableOp�1conv3d_6/kernel/Regularizer/Square/ReadVariableOp�1conv3d_7/kernel/Regularizer/Square/ReadVariableOp�.dense/kernel/Regularizer/Square/ReadVariableOp�model/StatefulPartitionedCall�model/StatefulPartitionedCall_1�
model/StatefulPartitionedCallStatefulPartitionedCallinputs_1model_14509model_14511model_14513model_14515model_14517model_14519model_14521model_14523model_14525model_14527model_14529model_14531model_14533model_14535model_14537model_14539model_14541model_14543*
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
@__inference_model_layer_call_and_return_conditional_losses_13738�
model/StatefulPartitionedCall_1StatefulPartitionedCallinputsmodel_14509model_14511model_14513model_14515model_14517model_14519model_14521model_14523model_14525model_14527model_14529model_14531model_14533model_14535model_14537model_14539model_14541model_14543*
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
@__inference_model_layer_call_and_return_conditional_losses_13738V
tf.compat.v1.squeeze_1/SqueezeSqueezeinputs_3*
T0*
_output_shapes
:t
tf.compat.v1.squeeze_3/SqueezeSqueeze&model/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:T
tf.compat.v1.squeeze/SqueezeSqueezeinputs_2*
T0*
_output_shapes
:v
tf.compat.v1.squeeze_2/SqueezeSqueeze(model/StatefulPartitionedCall_1:output:0*
T0*
_output_shapes
:v
tf.compat.v1.squeeze_4/SqueezeSqueeze(model/StatefulPartitionedCall_1:output:0*
T0*
_output_shapes
:t
tf.compat.v1.squeeze_5/SqueezeSqueeze&model/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:V
tf.compat.v1.squeeze_7/SqueezeSqueezeinputs_3*
T0*
_output_shapes
:t
tf.compat.v1.squeeze_9/SqueezeSqueeze&model/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:V
tf.compat.v1.squeeze_6/SqueezeSqueezeinputs_2*
T0*
_output_shapes
:v
tf.compat.v1.squeeze_8/SqueezeSqueeze(model/StatefulPartitionedCall_1:output:0*
T0*
_output_shapes
:u
tf.compat.v1.squeeze_13/SqueezeSqueeze&model/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:W
tf.compat.v1.squeeze_12/SqueezeSqueezeinputs_3*
T0*
_output_shapes
:w
tf.compat.v1.squeeze_11/SqueezeSqueeze(model/StatefulPartitionedCall_1:output:0*
T0*
_output_shapes
:W
tf.compat.v1.squeeze_10/SqueezeSqueezeinputs_2*
T0*
_output_shapes
:�
tf.math.subtract_1/SubSub'tf.compat.v1.squeeze_1/Squeeze:output:0'tf.compat.v1.squeeze_3/Squeeze:output:0*
T0*
_output_shapes
:�
tf.math.subtract/SubSub%tf.compat.v1.squeeze/Squeeze:output:0'tf.compat.v1.squeeze_2/Squeeze:output:0*
T0*
_output_shapes
:�
tf.math.subtract_2/SubSub'tf.compat.v1.squeeze_4/Squeeze:output:0'tf.compat.v1.squeeze_5/Squeeze:output:0*
T0*
_output_shapes
:�
tf.math.subtract_5/SubSub'tf.compat.v1.squeeze_7/Squeeze:output:0'tf.compat.v1.squeeze_9/Squeeze:output:0*
T0*
_output_shapes
:�
tf.math.subtract_4/SubSub'tf.compat.v1.squeeze_6/Squeeze:output:0'tf.compat.v1.squeeze_8/Squeeze:output:0*
T0*
_output_shapes
:m
tf.math.reduce_mean_8/RankRank(tf.compat.v1.squeeze_13/Squeeze:output:0*
T0*
_output_shapes
: c
!tf.math.reduce_mean_8/range/startConst*
_output_shapes
: *
dtype0*
value	B : c
!tf.math.reduce_mean_8/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
tf.math.reduce_mean_8/rangeRange*tf.math.reduce_mean_8/range/start:output:0#tf.math.reduce_mean_8/Rank:output:0*tf.math.reduce_mean_8/range/delta:output:0*#
_output_shapes
:����������
tf.math.reduce_mean_8/MeanMean(tf.compat.v1.squeeze_13/Squeeze:output:0$tf.math.reduce_mean_8/range:output:0*
T0*
_output_shapes
: m
tf.math.reduce_mean_7/RankRank(tf.compat.v1.squeeze_12/Squeeze:output:0*
T0*
_output_shapes
: c
!tf.math.reduce_mean_7/range/startConst*
_output_shapes
: *
dtype0*
value	B : c
!tf.math.reduce_mean_7/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
tf.math.reduce_mean_7/rangeRange*tf.math.reduce_mean_7/range/start:output:0#tf.math.reduce_mean_7/Rank:output:0*tf.math.reduce_mean_7/range/delta:output:0*#
_output_shapes
:����������
tf.math.reduce_mean_7/MeanMean(tf.compat.v1.squeeze_12/Squeeze:output:0$tf.math.reduce_mean_7/range:output:0*
T0*
_output_shapes
: m
tf.math.reduce_mean_6/RankRank(tf.compat.v1.squeeze_11/Squeeze:output:0*
T0*
_output_shapes
: c
!tf.math.reduce_mean_6/range/startConst*
_output_shapes
: *
dtype0*
value	B : c
!tf.math.reduce_mean_6/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
tf.math.reduce_mean_6/rangeRange*tf.math.reduce_mean_6/range/start:output:0#tf.math.reduce_mean_6/Rank:output:0*tf.math.reduce_mean_6/range/delta:output:0*#
_output_shapes
:����������
tf.math.reduce_mean_6/MeanMean(tf.compat.v1.squeeze_11/Squeeze:output:0$tf.math.reduce_mean_6/range:output:0*
T0*
_output_shapes
: m
tf.math.reduce_mean_5/RankRank(tf.compat.v1.squeeze_10/Squeeze:output:0*
T0*
_output_shapes
: c
!tf.math.reduce_mean_5/range/startConst*
_output_shapes
: *
dtype0*
value	B : c
!tf.math.reduce_mean_5/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
tf.math.reduce_mean_5/rangeRange*tf.math.reduce_mean_5/range/start:output:0#tf.math.reduce_mean_5/Rank:output:0*tf.math.reduce_mean_5/range/delta:output:0*#
_output_shapes
:����������
tf.math.reduce_mean_5/MeanMean(tf.compat.v1.squeeze_10/Squeeze:output:0$tf.math.reduce_mean_5/range:output:0*
T0*
_output_shapes
: `
tf.math.square_1/SquareSquaretf.math.subtract_1/Sub:z:0*
T0*
_output_shapes
:\
tf.math.square/SquareSquaretf.math.subtract/Sub:z:0*
T0*
_output_shapes
:`
tf.math.square_2/SquareSquaretf.math.subtract_2/Sub:z:0*
T0*
_output_shapes
:`
tf.math.square_4/SquareSquaretf.math.subtract_5/Sub:z:0*
T0*
_output_shapes
:`
tf.math.square_3/SquareSquaretf.math.subtract_4/Sub:z:0*
T0*
_output_shapes
:�
tf.math.subtract_10/SubSub(tf.compat.v1.squeeze_13/Squeeze:output:0#tf.math.reduce_mean_8/Mean:output:0*
T0*
_output_shapes
:�
tf.math.subtract_9/SubSub(tf.compat.v1.squeeze_12/Squeeze:output:0#tf.math.reduce_mean_7/Mean:output:0*
T0*
_output_shapes
:�
tf.math.subtract_7/SubSub(tf.compat.v1.squeeze_11/Squeeze:output:0#tf.math.reduce_mean_6/Mean:output:0*
T0*
_output_shapes
:�
tf.math.subtract_6/SubSub(tf.compat.v1.squeeze_10/Squeeze:output:0#tf.math.reduce_mean_5/Mean:output:0*
T0*
_output_shapes
:\
tf.math.reduce_mean/RankRanktf.math.square/Square:y:0*
T0*
_output_shapes
: a
tf.math.reduce_mean/range/startConst*
_output_shapes
: *
dtype0*
value	B : a
tf.math.reduce_mean/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
tf.math.reduce_mean/rangeRange(tf.math.reduce_mean/range/start:output:0!tf.math.reduce_mean/Rank:output:0(tf.math.reduce_mean/range/delta:output:0*#
_output_shapes
:����������
tf.math.reduce_mean/MeanMeantf.math.square/Square:y:0"tf.math.reduce_mean/range:output:0*
T0*
_output_shapes
: `
tf.math.reduce_mean_1/RankRanktf.math.square_1/Square:y:0*
T0*
_output_shapes
: c
!tf.math.reduce_mean_1/range/startConst*
_output_shapes
: *
dtype0*
value	B : c
!tf.math.reduce_mean_1/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
tf.math.reduce_mean_1/rangeRange*tf.math.reduce_mean_1/range/start:output:0#tf.math.reduce_mean_1/Rank:output:0*tf.math.reduce_mean_1/range/delta:output:0*#
_output_shapes
:����������
tf.math.reduce_mean_1/MeanMeantf.math.square_1/Square:y:0$tf.math.reduce_mean_1/range:output:0*
T0*
_output_shapes
: ^
tf.math.maximum/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *��:�
tf.math.maximum/MaximumMaximumtf.math.square_2/Square:y:0"tf.math.maximum/Maximum/y:output:0*
T0*
_output_shapes
:`
tf.math.reduce_mean_3/RankRanktf.math.square_3/Square:y:0*
T0*
_output_shapes
: c
!tf.math.reduce_mean_3/range/startConst*
_output_shapes
: *
dtype0*
value	B : c
!tf.math.reduce_mean_3/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
tf.math.reduce_mean_3/rangeRange*tf.math.reduce_mean_3/range/start:output:0#tf.math.reduce_mean_3/Rank:output:0*tf.math.reduce_mean_3/range/delta:output:0*#
_output_shapes
:����������
tf.math.reduce_mean_3/MeanMeantf.math.square_3/Square:y:0$tf.math.reduce_mean_3/range:output:0*
T0*
_output_shapes
: `
tf.math.reduce_mean_4/RankRanktf.math.square_4/Square:y:0*
T0*
_output_shapes
: c
!tf.math.reduce_mean_4/range/startConst*
_output_shapes
: *
dtype0*
value	B : c
!tf.math.reduce_mean_4/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
tf.math.reduce_mean_4/rangeRange*tf.math.reduce_mean_4/range/start:output:0#tf.math.reduce_mean_4/Rank:output:0*tf.math.reduce_mean_4/range/delta:output:0*#
_output_shapes
:����������
tf.math.reduce_mean_4/MeanMeantf.math.square_4/Square:y:0$tf.math.reduce_mean_4/range:output:0*
T0*
_output_shapes
: y
tf.math.multiply_3/MulMultf.math.subtract_9/Sub:z:0tf.math.subtract_10/Sub:z:0*
T0*
_output_shapes
:x
tf.math.multiply_1/MulMultf.math.subtract_6/Sub:z:0tf.math.subtract_7/Sub:z:0*
T0*
_output_shapes
:a
tf.math.square_8/SquareSquaretf.math.subtract_10/Sub:z:0*
T0*
_output_shapes
:`
tf.math.square_7/SquareSquaretf.math.subtract_9/Sub:z:0*
T0*
_output_shapes
:`
tf.math.square_6/SquareSquaretf.math.subtract_7/Sub:z:0*
T0*
_output_shapes
:`
tf.math.square_5/SquareSquaretf.math.subtract_6/Sub:z:0*
T0*
_output_shapes
:�
tf.__operators__.add/AddV2AddV2!tf.math.reduce_mean/Mean:output:0#tf.math.reduce_mean_1/Mean:output:0*
T0*
_output_shapes
: ]
tf.math.subtract_3/Sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *��:�
tf.math.subtract_3/SubSubtf.math.maximum/Maximum:z:0!tf.math.subtract_3/Sub/y:output:0*
T0*
_output_shapes
:�
tf.__operators__.add_2/AddV2AddV2#tf.math.reduce_mean_3/Mean:output:0#tf.math.reduce_mean_4/Mean:output:0*
T0*
_output_shapes
: ^
tf.math.reduce_sum_3/RankRanktf.math.multiply_3/Mul:z:0*
T0*
_output_shapes
: b
 tf.math.reduce_sum_3/range/startConst*
_output_shapes
: *
dtype0*
value	B : b
 tf.math.reduce_sum_3/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
tf.math.reduce_sum_3/rangeRange)tf.math.reduce_sum_3/range/start:output:0"tf.math.reduce_sum_3/Rank:output:0)tf.math.reduce_sum_3/range/delta:output:0*#
_output_shapes
:����������
tf.math.reduce_sum_3/SumSumtf.math.multiply_3/Mul:z:0#tf.math.reduce_sum_3/range:output:0*
T0*
_output_shapes
: \
tf.math.reduce_sum/RankRanktf.math.multiply_1/Mul:z:0*
T0*
_output_shapes
: `
tf.math.reduce_sum/range/startConst*
_output_shapes
: *
dtype0*
value	B : `
tf.math.reduce_sum/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
tf.math.reduce_sum/rangeRange'tf.math.reduce_sum/range/start:output:0 tf.math.reduce_sum/Rank:output:0'tf.math.reduce_sum/range/delta:output:0*#
_output_shapes
:���������}
tf.math.reduce_sum/SumSumtf.math.multiply_1/Mul:z:0!tf.math.reduce_sum/range:output:0*
T0*
_output_shapes
: _
tf.math.reduce_sum_4/RankRanktf.math.square_7/Square:y:0*
T0*
_output_shapes
: b
 tf.math.reduce_sum_4/range/startConst*
_output_shapes
: *
dtype0*
value	B : b
 tf.math.reduce_sum_4/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
tf.math.reduce_sum_4/rangeRange)tf.math.reduce_sum_4/range/start:output:0"tf.math.reduce_sum_4/Rank:output:0)tf.math.reduce_sum_4/range/delta:output:0*#
_output_shapes
:����������
tf.math.reduce_sum_4/SumSumtf.math.square_7/Square:y:0#tf.math.reduce_sum_4/range:output:0*
T0*
_output_shapes
: _
tf.math.reduce_sum_5/RankRanktf.math.square_8/Square:y:0*
T0*
_output_shapes
: b
 tf.math.reduce_sum_5/range/startConst*
_output_shapes
: *
dtype0*
value	B : b
 tf.math.reduce_sum_5/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
tf.math.reduce_sum_5/rangeRange)tf.math.reduce_sum_5/range/start:output:0"tf.math.reduce_sum_5/Rank:output:0)tf.math.reduce_sum_5/range/delta:output:0*#
_output_shapes
:����������
tf.math.reduce_sum_5/SumSumtf.math.square_8/Square:y:0#tf.math.reduce_sum_5/range:output:0*
T0*
_output_shapes
: _
tf.math.reduce_sum_1/RankRanktf.math.square_5/Square:y:0*
T0*
_output_shapes
: b
 tf.math.reduce_sum_1/range/startConst*
_output_shapes
: *
dtype0*
value	B : b
 tf.math.reduce_sum_1/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
tf.math.reduce_sum_1/rangeRange)tf.math.reduce_sum_1/range/start:output:0"tf.math.reduce_sum_1/Rank:output:0)tf.math.reduce_sum_1/range/delta:output:0*#
_output_shapes
:����������
tf.math.reduce_sum_1/SumSumtf.math.square_5/Square:y:0#tf.math.reduce_sum_1/range:output:0*
T0*
_output_shapes
: _
tf.math.reduce_sum_2/RankRanktf.math.square_6/Square:y:0*
T0*
_output_shapes
: b
 tf.math.reduce_sum_2/range/startConst*
_output_shapes
: *
dtype0*
value	B : b
 tf.math.reduce_sum_2/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
tf.math.reduce_sum_2/rangeRange)tf.math.reduce_sum_2/range/start:output:0"tf.math.reduce_sum_2/Rank:output:0)tf.math.reduce_sum_2/range/delta:output:0*#
_output_shapes
:����������
tf.math.reduce_sum_2/SumSumtf.math.square_6/Square:y:0#tf.math.reduce_sum_2/range:output:0*
T0*
_output_shapes
: _
tf.math.reduce_mean_2/RankRanktf.math.subtract_3/Sub:z:0*
T0*
_output_shapes
: c
!tf.math.reduce_mean_2/range/startConst*
_output_shapes
: *
dtype0*
value	B : c
!tf.math.reduce_mean_2/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
tf.math.reduce_mean_2/rangeRange*tf.math.reduce_mean_2/range/start:output:0#tf.math.reduce_mean_2/Rank:output:0*tf.math.reduce_mean_2/range/delta:output:0*#
_output_shapes
:����������
tf.math.reduce_mean_2/MeanMeantf.math.subtract_3/Sub:z:0$tf.math.reduce_mean_2/range:output:0*
T0*
_output_shapes
: �
"add_metric/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_2/AddV2:z:0add_metric_14680add_metric_14682*
Tin
2*
Tout
2*
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
GPU(2*0J 8� *N
fIRG
E__inference_add_metric_layer_call_and_return_conditional_losses_14235�
tf.math.multiply_4/MulMul!tf.math.reduce_sum_4/Sum:output:0!tf.math.reduce_sum_5/Sum:output:0*
T0*
_output_shapes
: �
tf.math.multiply_2/MulMul!tf.math.reduce_sum_1/Sum:output:0!tf.math.reduce_sum_2/Sum:output:0*
T0*
_output_shapes
: j
tf.math.multiply/MulMulunknown#tf.math.reduce_mean_2/Mean:output:0*
T0*
_output_shapes
: X
tf.math.sqrt_1/SqrtSqrttf.math.multiply_4/Mul:z:0*
T0*
_output_shapes
: V
tf.math.sqrt/SqrtSqrttf.math.multiply_2/Mul:z:0*
T0*
_output_shapes
: �
tf.__operators__.add_1/AddV2AddV2tf.__operators__.add/AddV2:z:0tf.math.multiply/Mul:z:0*
T0*
_output_shapes
: ]
tf.__operators__.add_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *��'7�
tf.__operators__.add_4/AddV2AddV2tf.math.sqrt_1/Sqrt:y:0!tf.__operators__.add_4/y:output:0*
T0*
_output_shapes
: ]
tf.__operators__.add_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *��'7�
tf.__operators__.add_3/AddV2AddV2tf.math.sqrt/Sqrt:y:0!tf.__operators__.add_3/y:output:0*
T0*
_output_shapes
: �
add_loss/PartitionedCallPartitionedCall tf.__operators__.add_1/AddV2:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: : * 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8� *L
fGRE
C__inference_add_loss_layer_call_and_return_conditional_losses_14258�
tf.math.truediv_1/truedivRealDiv!tf.math.reduce_sum_3/Sum:output:0 tf.__operators__.add_4/AddV2:z:0*
T0*
_output_shapes
: �
tf.math.truediv/truedivRealDivtf.math.reduce_sum/Sum:output:0 tf.__operators__.add_3/AddV2:z:0*
T0*
_output_shapes
: i
tf.math.subtract_11/SubSub	unknown_0tf.math.truediv_1/truediv:z:0*
T0*
_output_shapes
: f
tf.math.subtract_8/SubSub	unknown_1tf.math.truediv/truediv:z:0*
T0*
_output_shapes
: V
tf.math.pow/Pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?o
tf.math.pow/PowPowtf.math.subtract_8/Sub:z:0tf.math.pow/Pow/y:output:0*
T0*
_output_shapes
: X
tf.math.pow_1/Pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?t
tf.math.pow_1/PowPowtf.math.subtract_11/Sub:z:0tf.math.pow_1/Pow/y:output:0*
T0*
_output_shapes
: r
tf.__operators__.add_5/AddV2AddV2tf.math.pow/Pow:z:0tf.math.pow_1/Pow:z:0*
T0*
_output_shapes
: �
$add_metric_1/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_5/AddV2:z:0add_metric_1_14712add_metric_1_14714*
Tin
2*
Tout
2*
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
GPU(2*0J 8� *P
fKRI
G__inference_add_metric_1_layer_call_and_return_conditional_losses_14291�
/conv3d/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmodel_14509**
_output_shapes
:@*
dtype0�
 conv3d/kernel/Regularizer/SquareSquare7conv3d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0**
_output_shapes
:@|
conv3d/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
conv3d/kernel/Regularizer/SumSum$conv3d/kernel/Regularizer/Square:y:0(conv3d/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: d
conv3d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    �
conv3d/kernel/Regularizer/mulMul(conv3d/kernel/Regularizer/mul/x:output:0&conv3d/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1conv3d_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmodel_14513**
_output_shapes
:@@*
dtype0�
"conv3d_1/kernel/Regularizer/SquareSquare9conv3d_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0**
_output_shapes
:@@~
!conv3d_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
conv3d_1/kernel/Regularizer/SumSum&conv3d_1/kernel/Regularizer/Square:y:0*conv3d_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!conv3d_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    �
conv3d_1/kernel/Regularizer/mulMul*conv3d_1/kernel/Regularizer/mul/x:output:0(conv3d_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1conv3d_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmodel_14517*+
_output_shapes
:@�*
dtype0�
"conv3d_2/kernel/Regularizer/SquareSquare9conv3d_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*+
_output_shapes
:@�~
!conv3d_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
conv3d_2/kernel/Regularizer/SumSum&conv3d_2/kernel/Regularizer/Square:y:0*conv3d_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!conv3d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    �
conv3d_2/kernel/Regularizer/mulMul*conv3d_2/kernel/Regularizer/mul/x:output:0(conv3d_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1conv3d_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmodel_14521*,
_output_shapes
:��*
dtype0�
"conv3d_3/kernel/Regularizer/SquareSquare9conv3d_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*,
_output_shapes
:��~
!conv3d_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
conv3d_3/kernel/Regularizer/SumSum&conv3d_3/kernel/Regularizer/Square:y:0*conv3d_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!conv3d_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    �
conv3d_3/kernel/Regularizer/mulMul*conv3d_3/kernel/Regularizer/mul/x:output:0(conv3d_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1conv3d_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmodel_14525*,
_output_shapes
:��*
dtype0�
"conv3d_4/kernel/Regularizer/SquareSquare9conv3d_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*,
_output_shapes
:��~
!conv3d_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
conv3d_4/kernel/Regularizer/SumSum&conv3d_4/kernel/Regularizer/Square:y:0*conv3d_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!conv3d_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    �
conv3d_4/kernel/Regularizer/mulMul*conv3d_4/kernel/Regularizer/mul/x:output:0(conv3d_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1conv3d_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmodel_14529*,
_output_shapes
:��*
dtype0�
"conv3d_5/kernel/Regularizer/SquareSquare9conv3d_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*,
_output_shapes
:��~
!conv3d_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
conv3d_5/kernel/Regularizer/SumSum&conv3d_5/kernel/Regularizer/Square:y:0*conv3d_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!conv3d_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    �
conv3d_5/kernel/Regularizer/mulMul*conv3d_5/kernel/Regularizer/mul/x:output:0(conv3d_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1conv3d_6/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmodel_14533*,
_output_shapes
:��*
dtype0�
"conv3d_6/kernel/Regularizer/SquareSquare9conv3d_6/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*,
_output_shapes
:��~
!conv3d_6/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
conv3d_6/kernel/Regularizer/SumSum&conv3d_6/kernel/Regularizer/Square:y:0*conv3d_6/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!conv3d_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    �
conv3d_6/kernel/Regularizer/mulMul*conv3d_6/kernel/Regularizer/mul/x:output:0(conv3d_6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1conv3d_7/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmodel_14537*,
_output_shapes
:��*
dtype0�
"conv3d_7/kernel/Regularizer/SquareSquare9conv3d_7/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*,
_output_shapes
:��~
!conv3d_7/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
conv3d_7/kernel/Regularizer/SumSum&conv3d_7/kernel/Regularizer/Square:y:0*conv3d_7/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!conv3d_7/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    �
conv3d_7/kernel/Regularizer/mulMul*conv3d_7/kernel/Regularizer/mul/x:output:0(conv3d_7/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: {
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmodel_14541*
_output_shapes
:	�$*
dtype0�
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	�$o
dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
dense/kernel/Regularizer/SumSum#dense/kernel/Regularizer/Square:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: c
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    �
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: w
IdentityIdentity(model/StatefulPartitionedCall_1:output:0^NoOp*
T0*'
_output_shapes
:���������w

Identity_1Identity&model/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������a

Identity_2Identity!add_loss/PartitionedCall:output:1^NoOp*
T0*
_output_shapes
: �
NoOpNoOp#^add_metric/StatefulPartitionedCall%^add_metric_1/StatefulPartitionedCall0^conv3d/kernel/Regularizer/Square/ReadVariableOp2^conv3d_1/kernel/Regularizer/Square/ReadVariableOp2^conv3d_2/kernel/Regularizer/Square/ReadVariableOp2^conv3d_3/kernel/Regularizer/Square/ReadVariableOp2^conv3d_4/kernel/Regularizer/Square/ReadVariableOp2^conv3d_5/kernel/Regularizer/Square/ReadVariableOp2^conv3d_6/kernel/Regularizer/Square/ReadVariableOp2^conv3d_7/kernel/Regularizer/Square/ReadVariableOp/^dense/kernel/Regularizer/Square/ReadVariableOp^model/StatefulPartitionedCall ^model/StatefulPartitionedCall_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������22:���������22:���������:���������: : : : : : : : : : : : : : : : : : : : : : : : : 2H
"add_metric/StatefulPartitionedCall"add_metric/StatefulPartitionedCall2L
$add_metric_1/StatefulPartitionedCall$add_metric_1/StatefulPartitionedCall2b
/conv3d/kernel/Regularizer/Square/ReadVariableOp/conv3d/kernel/Regularizer/Square/ReadVariableOp2f
1conv3d_1/kernel/Regularizer/Square/ReadVariableOp1conv3d_1/kernel/Regularizer/Square/ReadVariableOp2f
1conv3d_2/kernel/Regularizer/Square/ReadVariableOp1conv3d_2/kernel/Regularizer/Square/ReadVariableOp2f
1conv3d_3/kernel/Regularizer/Square/ReadVariableOp1conv3d_3/kernel/Regularizer/Square/ReadVariableOp2f
1conv3d_4/kernel/Regularizer/Square/ReadVariableOp1conv3d_4/kernel/Regularizer/Square/ReadVariableOp2f
1conv3d_5/kernel/Regularizer/Square/ReadVariableOp1conv3d_5/kernel/Regularizer/Square/ReadVariableOp2f
1conv3d_6/kernel/Regularizer/Square/ReadVariableOp1conv3d_6/kernel/Regularizer/Square/ReadVariableOp2f
1conv3d_7/kernel/Regularizer/Square/ReadVariableOp1conv3d_7/kernel/Regularizer/Square/ReadVariableOp2`
.dense/kernel/Regularizer/Square/ReadVariableOp.dense/kernel/Regularizer/Square/ReadVariableOp2>
model/StatefulPartitionedCallmodel/StatefulPartitionedCall2B
model/StatefulPartitionedCall_1model/StatefulPartitionedCall_1:[ W
3
_output_shapes!
:���������22
 
_user_specified_nameinputs:[W
3
_output_shapes!
:���������22
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
�
@__inference_dense_layer_call_and_return_conditional_losses_13391

inputs1
matmul_readvariableop_resource:	�$-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�.dense/kernel/Regularizer/Square/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�$*
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
:����������
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�$*
dtype0�
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	�$o
dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
dense/kernel/Regularizer/SumSum#dense/kernel/Regularizer/Square:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: c
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    �
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp/^dense/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������$: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2`
.dense/kernel/Regularizer/Square/ReadVariableOp.dense/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:����������$
 
_user_specified_nameinputs
�
�
A__inference_conv3d_layer_call_and_return_conditional_losses_16902

inputs<
conv3d_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�Conv3D/ReadVariableOp�/conv3d/kernel/Regularizer/Square/ReadVariableOp�
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:@*
dtype0�
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������22@*
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
:���������22@\
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:���������22@�
/conv3d/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:@*
dtype0�
 conv3d/kernel/Regularizer/SquareSquare7conv3d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0**
_output_shapes
:@|
conv3d/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
conv3d/kernel/Regularizer/SumSum$conv3d/kernel/Regularizer/Square:y:0(conv3d/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: d
conv3d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    �
conv3d/kernel/Regularizer/mulMul(conv3d/kernel/Regularizer/mul/x:output:0&conv3d/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:���������22@�
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp0^conv3d/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������22: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp2b
/conv3d/kernel/Regularizer/Square/ReadVariableOp/conv3d/kernel/Regularizer/Square/ReadVariableOp:[ W
3
_output_shapes!
:���������22
 
_user_specified_nameinputs
�
o
C__inference_add_loss_layer_call_and_return_conditional_losses_14258

inputs
identity

identity_1=
IdentityIdentityinputs*
T0*
_output_shapes
: ?

Identity_1Identityinputs*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: :> :

_output_shapes
: 
 
_user_specified_nameinputs
�
�
%__inference_model_layer_call_fn_16561

inputs%
unknown:@
	unknown_0:@'
	unknown_1:@@
	unknown_2:@(
	unknown_3:@�
	unknown_4:	�)
	unknown_5:��
	unknown_6:	�)
	unknown_7:��
	unknown_8:	�)
	unknown_9:��

unknown_10:	�*

unknown_11:��

unknown_12:	�*

unknown_13:��

unknown_14:	�

unknown_15:	�$

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
@__inference_model_layer_call_and_return_conditional_losses_13738o
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
�
�
C__inference_conv3d_4_layer_call_and_return_conditional_losses_13290

inputs>
conv3d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv3D/ReadVariableOp�1conv3d_4/kernel/Regularizer/Square/ReadVariableOp�
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource*,
_output_shapes
:��*
dtype0�
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :����������*
paddingSAME*
strides	
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :����������]
ReluReluBiasAdd:output:0*
T0*4
_output_shapes"
 :�����������
1conv3d_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv3d_readvariableop_resource*,
_output_shapes
:��*
dtype0�
"conv3d_4/kernel/Regularizer/SquareSquare9conv3d_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*,
_output_shapes
:��~
!conv3d_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
conv3d_4/kernel/Regularizer/SumSum&conv3d_4/kernel/Regularizer/Square:y:0*conv3d_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!conv3d_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    �
conv3d_4/kernel/Regularizer/mulMul*conv3d_4/kernel/Regularizer/mul/x:output:0(conv3d_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: n
IdentityIdentityRelu:activations:0^NoOp*
T0*4
_output_shapes"
 :�����������
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp2^conv3d_4/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp2f
1conv3d_4/kernel/Regularizer/Square/ReadVariableOp1conv3d_4/kernel/Regularizer/Square/ReadVariableOp:\ X
4
_output_shapes"
 :����������
 
_user_specified_nameinputs
�
�
(__inference_conv3d_2_layer_call_fn_16947

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
 :���������
�*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *L
fGRE
C__inference_conv3d_2_layer_call_and_return_conditional_losses_13243|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :���������
�`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������
@: : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:���������
@
 
_user_specified_nameinputs
�
f
J__inference_max_pooling3d_1_layer_call_and_return_conditional_losses_17000

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
__inference_loss_fn_6_17237Z
:conv3d_6_kernel_regularizer_square_readvariableop_resource:��
identity��1conv3d_6/kernel/Regularizer/Square/ReadVariableOp�
1conv3d_6/kernel/Regularizer/Square/ReadVariableOpReadVariableOp:conv3d_6_kernel_regularizer_square_readvariableop_resource*,
_output_shapes
:��*
dtype0�
"conv3d_6/kernel/Regularizer/SquareSquare9conv3d_6/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*,
_output_shapes
:��~
!conv3d_6/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
conv3d_6/kernel/Regularizer/SumSum&conv3d_6/kernel/Regularizer/Square:y:0*conv3d_6/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!conv3d_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    �
conv3d_6/kernel/Regularizer/mulMul*conv3d_6/kernel/Regularizer/mul/x:output:0(conv3d_6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: a
IdentityIdentity#conv3d_6/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: z
NoOpNoOp2^conv3d_6/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2f
1conv3d_6/kernel/Regularizer/Square/ReadVariableOp1conv3d_6/kernel/Regularizer/Square/ReadVariableOp
�
�
C__inference_conv3d_4_layer_call_and_return_conditional_losses_17026

inputs>
conv3d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv3D/ReadVariableOp�1conv3d_4/kernel/Regularizer/Square/ReadVariableOp�
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource*,
_output_shapes
:��*
dtype0�
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :����������*
paddingSAME*
strides	
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :����������]
ReluReluBiasAdd:output:0*
T0*4
_output_shapes"
 :�����������
1conv3d_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv3d_readvariableop_resource*,
_output_shapes
:��*
dtype0�
"conv3d_4/kernel/Regularizer/SquareSquare9conv3d_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*,
_output_shapes
:��~
!conv3d_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
conv3d_4/kernel/Regularizer/SumSum&conv3d_4/kernel/Regularizer/Square:y:0*conv3d_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!conv3d_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    �
conv3d_4/kernel/Regularizer/mulMul*conv3d_4/kernel/Regularizer/mul/x:output:0(conv3d_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: n
IdentityIdentityRelu:activations:0^NoOp*
T0*4
_output_shapes"
 :�����������
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp2^conv3d_4/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp2f
1conv3d_4/kernel/Regularizer/Square/ReadVariableOp1conv3d_4/kernel/Regularizer/Square/ReadVariableOp:\ X
4
_output_shapes"
 :����������
 
_user_specified_nameinputs
�
�
C__inference_conv3d_2_layer_call_and_return_conditional_losses_13243

inputs=
conv3d_readvariableop_resource:@�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv3D/ReadVariableOp�1conv3d_2/kernel/Regularizer/Square/ReadVariableOp�
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource*+
_output_shapes
:@�*
dtype0�
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :���������
�*
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
 :���������
�]
ReluReluBiasAdd:output:0*
T0*4
_output_shapes"
 :���������
��
1conv3d_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv3d_readvariableop_resource*+
_output_shapes
:@�*
dtype0�
"conv3d_2/kernel/Regularizer/SquareSquare9conv3d_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*+
_output_shapes
:@�~
!conv3d_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
conv3d_2/kernel/Regularizer/SumSum&conv3d_2/kernel/Regularizer/Square:y:0*conv3d_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!conv3d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    �
conv3d_2/kernel/Regularizer/mulMul*conv3d_2/kernel/Regularizer/mul/x:output:0(conv3d_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: n
IdentityIdentityRelu:activations:0^NoOp*
T0*4
_output_shapes"
 :���������
��
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp2^conv3d_2/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������
@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp2f
1conv3d_2/kernel/Regularizer/Square/ReadVariableOp1conv3d_2/kernel/Regularizer/Square/ReadVariableOp:[ W
3
_output_shapes!
:���������
@
 
_user_specified_nameinputs
�
�
%__inference_model_layer_call_fn_13491
input_1%
unknown:@
	unknown_0:@'
	unknown_1:@@
	unknown_2:@(
	unknown_3:@�
	unknown_4:	�)
	unknown_5:��
	unknown_6:	�)
	unknown_7:��
	unknown_8:	�)
	unknown_9:��

unknown_10:	�*

unknown_11:��

unknown_12:	�*

unknown_13:��

unknown_14:	�

unknown_15:	�$

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
@__inference_model_layer_call_and_return_conditional_losses_13452o
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
�
�
C__inference_conv3d_3_layer_call_and_return_conditional_losses_16990

inputs>
conv3d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv3D/ReadVariableOp�1conv3d_3/kernel/Regularizer/Square/ReadVariableOp�
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource*,
_output_shapes
:��*
dtype0�
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :���������
�*
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
 :���������
�]
ReluReluBiasAdd:output:0*
T0*4
_output_shapes"
 :���������
��
1conv3d_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv3d_readvariableop_resource*,
_output_shapes
:��*
dtype0�
"conv3d_3/kernel/Regularizer/SquareSquare9conv3d_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*,
_output_shapes
:��~
!conv3d_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
conv3d_3/kernel/Regularizer/SumSum&conv3d_3/kernel/Regularizer/Square:y:0*conv3d_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!conv3d_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    �
conv3d_3/kernel/Regularizer/mulMul*conv3d_3/kernel/Regularizer/mul/x:output:0(conv3d_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: n
IdentityIdentityRelu:activations:0^NoOp*
T0*4
_output_shapes"
 :���������
��
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp2^conv3d_3/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:���������
�: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp2f
1conv3d_3/kernel/Regularizer/Square/ReadVariableOp1conv3d_3/kernel/Regularizer/Square/ReadVariableOp:\ X
4
_output_shapes"
 :���������
�
 
_user_specified_nameinputs
�
f
J__inference_max_pooling3d_2_layer_call_and_return_conditional_losses_17062

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
��
�
@__inference_model_layer_call_and_return_conditional_losses_16687

inputsC
%conv3d_conv3d_readvariableop_resource:@4
&conv3d_biasadd_readvariableop_resource:@E
'conv3d_1_conv3d_readvariableop_resource:@@6
(conv3d_1_biasadd_readvariableop_resource:@F
'conv3d_2_conv3d_readvariableop_resource:@�7
(conv3d_2_biasadd_readvariableop_resource:	�G
'conv3d_3_conv3d_readvariableop_resource:��7
(conv3d_3_biasadd_readvariableop_resource:	�G
'conv3d_4_conv3d_readvariableop_resource:��7
(conv3d_4_biasadd_readvariableop_resource:	�G
'conv3d_5_conv3d_readvariableop_resource:��7
(conv3d_5_biasadd_readvariableop_resource:	�G
'conv3d_6_conv3d_readvariableop_resource:��7
(conv3d_6_biasadd_readvariableop_resource:	�G
'conv3d_7_conv3d_readvariableop_resource:��7
(conv3d_7_biasadd_readvariableop_resource:	�7
$dense_matmul_readvariableop_resource:	�$3
%dense_biasadd_readvariableop_resource:
identity��conv3d/BiasAdd/ReadVariableOp�conv3d/Conv3D/ReadVariableOp�/conv3d/kernel/Regularizer/Square/ReadVariableOp�conv3d_1/BiasAdd/ReadVariableOp�conv3d_1/Conv3D/ReadVariableOp�1conv3d_1/kernel/Regularizer/Square/ReadVariableOp�conv3d_2/BiasAdd/ReadVariableOp�conv3d_2/Conv3D/ReadVariableOp�1conv3d_2/kernel/Regularizer/Square/ReadVariableOp�conv3d_3/BiasAdd/ReadVariableOp�conv3d_3/Conv3D/ReadVariableOp�1conv3d_3/kernel/Regularizer/Square/ReadVariableOp�conv3d_4/BiasAdd/ReadVariableOp�conv3d_4/Conv3D/ReadVariableOp�1conv3d_4/kernel/Regularizer/Square/ReadVariableOp�conv3d_5/BiasAdd/ReadVariableOp�conv3d_5/Conv3D/ReadVariableOp�1conv3d_5/kernel/Regularizer/Square/ReadVariableOp�conv3d_6/BiasAdd/ReadVariableOp�conv3d_6/Conv3D/ReadVariableOp�1conv3d_6/kernel/Regularizer/Square/ReadVariableOp�conv3d_7/BiasAdd/ReadVariableOp�conv3d_7/Conv3D/ReadVariableOp�1conv3d_7/kernel/Regularizer/Square/ReadVariableOp�dense/BiasAdd/ReadVariableOp�dense/MatMul/ReadVariableOp�.dense/kernel/Regularizer/Square/ReadVariableOp�
conv3d/Conv3D/ReadVariableOpReadVariableOp%conv3d_conv3d_readvariableop_resource**
_output_shapes
:@*
dtype0�
conv3d/Conv3DConv3Dinputs$conv3d/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������22@*
paddingSAME*
strides	
�
conv3d/BiasAdd/ReadVariableOpReadVariableOp&conv3d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
conv3d/BiasAddBiasAddconv3d/Conv3D:output:0%conv3d/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������22@j
conv3d/ReluReluconv3d/BiasAdd:output:0*
T0*3
_output_shapes!
:���������22@�
conv3d_1/Conv3D/ReadVariableOpReadVariableOp'conv3d_1_conv3d_readvariableop_resource**
_output_shapes
:@@*
dtype0�
conv3d_1/Conv3DConv3Dconv3d/Relu:activations:0&conv3d_1/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������22@*
paddingSAME*
strides	
�
conv3d_1/BiasAdd/ReadVariableOpReadVariableOp(conv3d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
conv3d_1/BiasAddBiasAddconv3d_1/Conv3D:output:0'conv3d_1/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������22@n
conv3d_1/ReluReluconv3d_1/BiasAdd:output:0*
T0*3
_output_shapes!
:���������22@�
max_pooling3d/MaxPool3D	MaxPool3Dconv3d_1/Relu:activations:0*
T0*3
_output_shapes!
:���������
@*
ksize	
*
paddingVALID*
strides	
�
conv3d_2/Conv3D/ReadVariableOpReadVariableOp'conv3d_2_conv3d_readvariableop_resource*+
_output_shapes
:@�*
dtype0�
conv3d_2/Conv3DConv3D max_pooling3d/MaxPool3D:output:0&conv3d_2/Conv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :���������
�*
paddingSAME*
strides	
�
conv3d_2/BiasAdd/ReadVariableOpReadVariableOp(conv3d_2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
conv3d_2/BiasAddBiasAddconv3d_2/Conv3D:output:0'conv3d_2/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :���������
�o
conv3d_2/ReluReluconv3d_2/BiasAdd:output:0*
T0*4
_output_shapes"
 :���������
��
conv3d_3/Conv3D/ReadVariableOpReadVariableOp'conv3d_3_conv3d_readvariableop_resource*,
_output_shapes
:��*
dtype0�
conv3d_3/Conv3DConv3Dconv3d_2/Relu:activations:0&conv3d_3/Conv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :���������
�*
paddingSAME*
strides	
�
conv3d_3/BiasAdd/ReadVariableOpReadVariableOp(conv3d_3_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
conv3d_3/BiasAddBiasAddconv3d_3/Conv3D:output:0'conv3d_3/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :���������
�o
conv3d_3/ReluReluconv3d_3/BiasAdd:output:0*
T0*4
_output_shapes"
 :���������
��
max_pooling3d_1/MaxPool3D	MaxPool3Dconv3d_3/Relu:activations:0*
T0*4
_output_shapes"
 :����������*
ksize	
*
paddingVALID*
strides	
�
conv3d_4/Conv3D/ReadVariableOpReadVariableOp'conv3d_4_conv3d_readvariableop_resource*,
_output_shapes
:��*
dtype0�
conv3d_4/Conv3DConv3D"max_pooling3d_1/MaxPool3D:output:0&conv3d_4/Conv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :����������*
paddingSAME*
strides	
�
conv3d_4/BiasAdd/ReadVariableOpReadVariableOp(conv3d_4_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
conv3d_4/BiasAddBiasAddconv3d_4/Conv3D:output:0'conv3d_4/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :����������o
conv3d_4/ReluReluconv3d_4/BiasAdd:output:0*
T0*4
_output_shapes"
 :�����������
conv3d_5/Conv3D/ReadVariableOpReadVariableOp'conv3d_5_conv3d_readvariableop_resource*,
_output_shapes
:��*
dtype0�
conv3d_5/Conv3DConv3Dconv3d_4/Relu:activations:0&conv3d_5/Conv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :����������*
paddingSAME*
strides	
�
conv3d_5/BiasAdd/ReadVariableOpReadVariableOp(conv3d_5_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
conv3d_5/BiasAddBiasAddconv3d_5/Conv3D:output:0'conv3d_5/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :����������o
conv3d_5/ReluReluconv3d_5/BiasAdd:output:0*
T0*4
_output_shapes"
 :�����������
max_pooling3d_2/MaxPool3D	MaxPool3Dconv3d_5/Relu:activations:0*
T0*4
_output_shapes"
 :����������*
ksize	
*
paddingVALID*
strides	
�
conv3d_6/Conv3D/ReadVariableOpReadVariableOp'conv3d_6_conv3d_readvariableop_resource*,
_output_shapes
:��*
dtype0�
conv3d_6/Conv3DConv3D"max_pooling3d_2/MaxPool3D:output:0&conv3d_6/Conv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :����������*
paddingSAME*
strides	
�
conv3d_6/BiasAdd/ReadVariableOpReadVariableOp(conv3d_6_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
conv3d_6/BiasAddBiasAddconv3d_6/Conv3D:output:0'conv3d_6/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :����������o
conv3d_6/ReluReluconv3d_6/BiasAdd:output:0*
T0*4
_output_shapes"
 :�����������
conv3d_7/Conv3D/ReadVariableOpReadVariableOp'conv3d_7_conv3d_readvariableop_resource*,
_output_shapes
:��*
dtype0�
conv3d_7/Conv3DConv3Dconv3d_6/Relu:activations:0&conv3d_7/Conv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :����������*
paddingSAME*
strides	
�
conv3d_7/BiasAdd/ReadVariableOpReadVariableOp(conv3d_7_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
conv3d_7/BiasAddBiasAddconv3d_7/Conv3D:output:0'conv3d_7/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :����������o
conv3d_7/ReluReluconv3d_7/BiasAdd:output:0*
T0*4
_output_shapes"
 :�����������
max_pooling3d_3/MaxPool3D	MaxPool3Dconv3d_7/Relu:activations:0*
T0*4
_output_shapes"
 :����������*
ksize	
*
paddingVALID*
strides	
^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   �
flatten/ReshapeReshape"max_pooling3d_3/MaxPool3D:output:0flatten/Const:output:0*
T0*(
_output_shapes
:����������$�
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	�$*
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
/conv3d/kernel/Regularizer/Square/ReadVariableOpReadVariableOp%conv3d_conv3d_readvariableop_resource**
_output_shapes
:@*
dtype0�
 conv3d/kernel/Regularizer/SquareSquare7conv3d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0**
_output_shapes
:@|
conv3d/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
conv3d/kernel/Regularizer/SumSum$conv3d/kernel/Regularizer/Square:y:0(conv3d/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: d
conv3d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    �
conv3d/kernel/Regularizer/mulMul(conv3d/kernel/Regularizer/mul/x:output:0&conv3d/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1conv3d_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv3d_1_conv3d_readvariableop_resource**
_output_shapes
:@@*
dtype0�
"conv3d_1/kernel/Regularizer/SquareSquare9conv3d_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0**
_output_shapes
:@@~
!conv3d_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
conv3d_1/kernel/Regularizer/SumSum&conv3d_1/kernel/Regularizer/Square:y:0*conv3d_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!conv3d_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    �
conv3d_1/kernel/Regularizer/mulMul*conv3d_1/kernel/Regularizer/mul/x:output:0(conv3d_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1conv3d_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv3d_2_conv3d_readvariableop_resource*+
_output_shapes
:@�*
dtype0�
"conv3d_2/kernel/Regularizer/SquareSquare9conv3d_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*+
_output_shapes
:@�~
!conv3d_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
conv3d_2/kernel/Regularizer/SumSum&conv3d_2/kernel/Regularizer/Square:y:0*conv3d_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!conv3d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    �
conv3d_2/kernel/Regularizer/mulMul*conv3d_2/kernel/Regularizer/mul/x:output:0(conv3d_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1conv3d_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv3d_3_conv3d_readvariableop_resource*,
_output_shapes
:��*
dtype0�
"conv3d_3/kernel/Regularizer/SquareSquare9conv3d_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*,
_output_shapes
:��~
!conv3d_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
conv3d_3/kernel/Regularizer/SumSum&conv3d_3/kernel/Regularizer/Square:y:0*conv3d_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!conv3d_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    �
conv3d_3/kernel/Regularizer/mulMul*conv3d_3/kernel/Regularizer/mul/x:output:0(conv3d_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1conv3d_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv3d_4_conv3d_readvariableop_resource*,
_output_shapes
:��*
dtype0�
"conv3d_4/kernel/Regularizer/SquareSquare9conv3d_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*,
_output_shapes
:��~
!conv3d_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
conv3d_4/kernel/Regularizer/SumSum&conv3d_4/kernel/Regularizer/Square:y:0*conv3d_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!conv3d_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    �
conv3d_4/kernel/Regularizer/mulMul*conv3d_4/kernel/Regularizer/mul/x:output:0(conv3d_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1conv3d_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv3d_5_conv3d_readvariableop_resource*,
_output_shapes
:��*
dtype0�
"conv3d_5/kernel/Regularizer/SquareSquare9conv3d_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*,
_output_shapes
:��~
!conv3d_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
conv3d_5/kernel/Regularizer/SumSum&conv3d_5/kernel/Regularizer/Square:y:0*conv3d_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!conv3d_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    �
conv3d_5/kernel/Regularizer/mulMul*conv3d_5/kernel/Regularizer/mul/x:output:0(conv3d_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1conv3d_6/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv3d_6_conv3d_readvariableop_resource*,
_output_shapes
:��*
dtype0�
"conv3d_6/kernel/Regularizer/SquareSquare9conv3d_6/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*,
_output_shapes
:��~
!conv3d_6/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
conv3d_6/kernel/Regularizer/SumSum&conv3d_6/kernel/Regularizer/Square:y:0*conv3d_6/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!conv3d_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    �
conv3d_6/kernel/Regularizer/mulMul*conv3d_6/kernel/Regularizer/mul/x:output:0(conv3d_6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1conv3d_7/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv3d_7_conv3d_readvariableop_resource*,
_output_shapes
:��*
dtype0�
"conv3d_7/kernel/Regularizer/SquareSquare9conv3d_7/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*,
_output_shapes
:��~
!conv3d_7/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
conv3d_7/kernel/Regularizer/SumSum&conv3d_7/kernel/Regularizer/Square:y:0*conv3d_7/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!conv3d_7/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    �
conv3d_7/kernel/Regularizer/mulMul*conv3d_7/kernel/Regularizer/mul/x:output:0(conv3d_7/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	�$*
dtype0�
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	�$o
dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
dense/kernel/Regularizer/SumSum#dense/kernel/Regularizer/Square:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: c
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    �
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: e
IdentityIdentitydense/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^conv3d/BiasAdd/ReadVariableOp^conv3d/Conv3D/ReadVariableOp0^conv3d/kernel/Regularizer/Square/ReadVariableOp ^conv3d_1/BiasAdd/ReadVariableOp^conv3d_1/Conv3D/ReadVariableOp2^conv3d_1/kernel/Regularizer/Square/ReadVariableOp ^conv3d_2/BiasAdd/ReadVariableOp^conv3d_2/Conv3D/ReadVariableOp2^conv3d_2/kernel/Regularizer/Square/ReadVariableOp ^conv3d_3/BiasAdd/ReadVariableOp^conv3d_3/Conv3D/ReadVariableOp2^conv3d_3/kernel/Regularizer/Square/ReadVariableOp ^conv3d_4/BiasAdd/ReadVariableOp^conv3d_4/Conv3D/ReadVariableOp2^conv3d_4/kernel/Regularizer/Square/ReadVariableOp ^conv3d_5/BiasAdd/ReadVariableOp^conv3d_5/Conv3D/ReadVariableOp2^conv3d_5/kernel/Regularizer/Square/ReadVariableOp ^conv3d_6/BiasAdd/ReadVariableOp^conv3d_6/Conv3D/ReadVariableOp2^conv3d_6/kernel/Regularizer/Square/ReadVariableOp ^conv3d_7/BiasAdd/ReadVariableOp^conv3d_7/Conv3D/ReadVariableOp2^conv3d_7/kernel/Regularizer/Square/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp/^dense/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:���������22: : : : : : : : : : : : : : : : : : 2>
conv3d/BiasAdd/ReadVariableOpconv3d/BiasAdd/ReadVariableOp2<
conv3d/Conv3D/ReadVariableOpconv3d/Conv3D/ReadVariableOp2b
/conv3d/kernel/Regularizer/Square/ReadVariableOp/conv3d/kernel/Regularizer/Square/ReadVariableOp2B
conv3d_1/BiasAdd/ReadVariableOpconv3d_1/BiasAdd/ReadVariableOp2@
conv3d_1/Conv3D/ReadVariableOpconv3d_1/Conv3D/ReadVariableOp2f
1conv3d_1/kernel/Regularizer/Square/ReadVariableOp1conv3d_1/kernel/Regularizer/Square/ReadVariableOp2B
conv3d_2/BiasAdd/ReadVariableOpconv3d_2/BiasAdd/ReadVariableOp2@
conv3d_2/Conv3D/ReadVariableOpconv3d_2/Conv3D/ReadVariableOp2f
1conv3d_2/kernel/Regularizer/Square/ReadVariableOp1conv3d_2/kernel/Regularizer/Square/ReadVariableOp2B
conv3d_3/BiasAdd/ReadVariableOpconv3d_3/BiasAdd/ReadVariableOp2@
conv3d_3/Conv3D/ReadVariableOpconv3d_3/Conv3D/ReadVariableOp2f
1conv3d_3/kernel/Regularizer/Square/ReadVariableOp1conv3d_3/kernel/Regularizer/Square/ReadVariableOp2B
conv3d_4/BiasAdd/ReadVariableOpconv3d_4/BiasAdd/ReadVariableOp2@
conv3d_4/Conv3D/ReadVariableOpconv3d_4/Conv3D/ReadVariableOp2f
1conv3d_4/kernel/Regularizer/Square/ReadVariableOp1conv3d_4/kernel/Regularizer/Square/ReadVariableOp2B
conv3d_5/BiasAdd/ReadVariableOpconv3d_5/BiasAdd/ReadVariableOp2@
conv3d_5/Conv3D/ReadVariableOpconv3d_5/Conv3D/ReadVariableOp2f
1conv3d_5/kernel/Regularizer/Square/ReadVariableOp1conv3d_5/kernel/Regularizer/Square/ReadVariableOp2B
conv3d_6/BiasAdd/ReadVariableOpconv3d_6/BiasAdd/ReadVariableOp2@
conv3d_6/Conv3D/ReadVariableOpconv3d_6/Conv3D/ReadVariableOp2f
1conv3d_6/kernel/Regularizer/Square/ReadVariableOp1conv3d_6/kernel/Regularizer/Square/ReadVariableOp2B
conv3d_7/BiasAdd/ReadVariableOpconv3d_7/BiasAdd/ReadVariableOp2@
conv3d_7/Conv3D/ReadVariableOpconv3d_7/Conv3D/ReadVariableOp2f
1conv3d_7/kernel/Regularizer/Square/ReadVariableOp1conv3d_7/kernel/Regularizer/Square/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2`
.dense/kernel/Regularizer/Square/ReadVariableOp.dense/kernel/Regularizer/Square/ReadVariableOp:[ W
3
_output_shapes!
:���������22
 
_user_specified_nameinputs
�
�
(__inference_conv3d_7_layer_call_fn_17097

inputs'
unknown:��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :����������*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *L
fGRE
C__inference_conv3d_7_layer_call_and_return_conditional_losses_13360|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :����������
 
_user_specified_nameinputs
�
f
J__inference_max_pooling3d_3_layer_call_and_return_conditional_losses_13169

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
%__inference_model_layer_call_fn_13818
input_1%
unknown:@
	unknown_0:@'
	unknown_1:@@
	unknown_2:@(
	unknown_3:@�
	unknown_4:	�)
	unknown_5:��
	unknown_6:	�)
	unknown_7:��
	unknown_8:	�)
	unknown_9:��

unknown_10:	�*

unknown_11:��

unknown_12:	�*

unknown_13:��

unknown_14:	�

unknown_15:	�$

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
@__inference_model_layer_call_and_return_conditional_losses_13738o
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
�
K
/__inference_max_pooling3d_3_layer_call_fn_17119

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
J__inference_max_pooling3d_3_layer_call_and_return_conditional_losses_13169�
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
�
D
(__inference_add_loss_layer_call_fn_16819

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: : * 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8� *L
fGRE
C__inference_add_loss_layer_call_and_return_conditional_losses_14258O
IdentityIdentityPartitionedCall:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: :> :

_output_shapes
: 
 
_user_specified_nameinputs
�
f
J__inference_max_pooling3d_3_layer_call_and_return_conditional_losses_17124

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
�
�
(__inference_conv3d_4_layer_call_fn_17009

inputs'
unknown:��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :����������*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *L
fGRE
C__inference_conv3d_4_layer_call_and_return_conditional_losses_13290|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :����������
 
_user_specified_nameinputs
�
�
__inference_loss_fn_2_17193Y
:conv3d_2_kernel_regularizer_square_readvariableop_resource:@�
identity��1conv3d_2/kernel/Regularizer/Square/ReadVariableOp�
1conv3d_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp:conv3d_2_kernel_regularizer_square_readvariableop_resource*+
_output_shapes
:@�*
dtype0�
"conv3d_2/kernel/Regularizer/SquareSquare9conv3d_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*+
_output_shapes
:@�~
!conv3d_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
conv3d_2/kernel/Regularizer/SumSum&conv3d_2/kernel/Regularizer/Square:y:0*conv3d_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!conv3d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    �
conv3d_2/kernel/Regularizer/mulMul*conv3d_2/kernel/Regularizer/mul/x:output:0(conv3d_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: a
IdentityIdentity#conv3d_2/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: z
NoOpNoOp2^conv3d_2/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2f
1conv3d_2/kernel/Regularizer/Square/ReadVariableOp1conv3d_2/kernel/Regularizer/Square/ReadVariableOp
��
�
B__inference_model_1_layer_call_and_return_conditional_losses_16425
inputs_0
inputs_1
inputs_2
inputs_3I
+model_conv3d_conv3d_readvariableop_resource:@:
,model_conv3d_biasadd_readvariableop_resource:@K
-model_conv3d_1_conv3d_readvariableop_resource:@@<
.model_conv3d_1_biasadd_readvariableop_resource:@L
-model_conv3d_2_conv3d_readvariableop_resource:@�=
.model_conv3d_2_biasadd_readvariableop_resource:	�M
-model_conv3d_3_conv3d_readvariableop_resource:��=
.model_conv3d_3_biasadd_readvariableop_resource:	�M
-model_conv3d_4_conv3d_readvariableop_resource:��=
.model_conv3d_4_biasadd_readvariableop_resource:	�M
-model_conv3d_5_conv3d_readvariableop_resource:��=
.model_conv3d_5_biasadd_readvariableop_resource:	�M
-model_conv3d_6_conv3d_readvariableop_resource:��=
.model_conv3d_6_biasadd_readvariableop_resource:	�M
-model_conv3d_7_conv3d_readvariableop_resource:��=
.model_conv3d_7_biasadd_readvariableop_resource:	�=
*model_dense_matmul_readvariableop_resource:	�$9
+model_dense_biasadd_readvariableop_resource:1
'add_metric_assignaddvariableop_resource: 3
)add_metric_assignaddvariableop_1_resource: 
unknown
	unknown_0
	unknown_13
)add_metric_1_assignaddvariableop_resource: 5
+add_metric_1_assignaddvariableop_1_resource: 
identity

identity_1

identity_2��add_metric/AssignAddVariableOp� add_metric/AssignAddVariableOp_1�$add_metric/div_no_nan/ReadVariableOp�&add_metric/div_no_nan/ReadVariableOp_1� add_metric_1/AssignAddVariableOp�"add_metric_1/AssignAddVariableOp_1�&add_metric_1/div_no_nan/ReadVariableOp�(add_metric_1/div_no_nan/ReadVariableOp_1�/conv3d/kernel/Regularizer/Square/ReadVariableOp�1conv3d_1/kernel/Regularizer/Square/ReadVariableOp�1conv3d_2/kernel/Regularizer/Square/ReadVariableOp�1conv3d_3/kernel/Regularizer/Square/ReadVariableOp�1conv3d_4/kernel/Regularizer/Square/ReadVariableOp�1conv3d_5/kernel/Regularizer/Square/ReadVariableOp�1conv3d_6/kernel/Regularizer/Square/ReadVariableOp�1conv3d_7/kernel/Regularizer/Square/ReadVariableOp�.dense/kernel/Regularizer/Square/ReadVariableOp�#model/conv3d/BiasAdd/ReadVariableOp�%model/conv3d/BiasAdd_1/ReadVariableOp�"model/conv3d/Conv3D/ReadVariableOp�$model/conv3d/Conv3D_1/ReadVariableOp�%model/conv3d_1/BiasAdd/ReadVariableOp�'model/conv3d_1/BiasAdd_1/ReadVariableOp�$model/conv3d_1/Conv3D/ReadVariableOp�&model/conv3d_1/Conv3D_1/ReadVariableOp�%model/conv3d_2/BiasAdd/ReadVariableOp�'model/conv3d_2/BiasAdd_1/ReadVariableOp�$model/conv3d_2/Conv3D/ReadVariableOp�&model/conv3d_2/Conv3D_1/ReadVariableOp�%model/conv3d_3/BiasAdd/ReadVariableOp�'model/conv3d_3/BiasAdd_1/ReadVariableOp�$model/conv3d_3/Conv3D/ReadVariableOp�&model/conv3d_3/Conv3D_1/ReadVariableOp�%model/conv3d_4/BiasAdd/ReadVariableOp�'model/conv3d_4/BiasAdd_1/ReadVariableOp�$model/conv3d_4/Conv3D/ReadVariableOp�&model/conv3d_4/Conv3D_1/ReadVariableOp�%model/conv3d_5/BiasAdd/ReadVariableOp�'model/conv3d_5/BiasAdd_1/ReadVariableOp�$model/conv3d_5/Conv3D/ReadVariableOp�&model/conv3d_5/Conv3D_1/ReadVariableOp�%model/conv3d_6/BiasAdd/ReadVariableOp�'model/conv3d_6/BiasAdd_1/ReadVariableOp�$model/conv3d_6/Conv3D/ReadVariableOp�&model/conv3d_6/Conv3D_1/ReadVariableOp�%model/conv3d_7/BiasAdd/ReadVariableOp�'model/conv3d_7/BiasAdd_1/ReadVariableOp�$model/conv3d_7/Conv3D/ReadVariableOp�&model/conv3d_7/Conv3D_1/ReadVariableOp�"model/dense/BiasAdd/ReadVariableOp�$model/dense/BiasAdd_1/ReadVariableOp�!model/dense/MatMul/ReadVariableOp�#model/dense/MatMul_1/ReadVariableOp�
"model/conv3d/Conv3D/ReadVariableOpReadVariableOp+model_conv3d_conv3d_readvariableop_resource**
_output_shapes
:@*
dtype0�
model/conv3d/Conv3DConv3Dinputs_1*model/conv3d/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������22@*
paddingSAME*
strides	
�
#model/conv3d/BiasAdd/ReadVariableOpReadVariableOp,model_conv3d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
model/conv3d/BiasAddBiasAddmodel/conv3d/Conv3D:output:0+model/conv3d/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������22@v
model/conv3d/ReluRelumodel/conv3d/BiasAdd:output:0*
T0*3
_output_shapes!
:���������22@�
$model/conv3d_1/Conv3D/ReadVariableOpReadVariableOp-model_conv3d_1_conv3d_readvariableop_resource**
_output_shapes
:@@*
dtype0�
model/conv3d_1/Conv3DConv3Dmodel/conv3d/Relu:activations:0,model/conv3d_1/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������22@*
paddingSAME*
strides	
�
%model/conv3d_1/BiasAdd/ReadVariableOpReadVariableOp.model_conv3d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
model/conv3d_1/BiasAddBiasAddmodel/conv3d_1/Conv3D:output:0-model/conv3d_1/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������22@z
model/conv3d_1/ReluRelumodel/conv3d_1/BiasAdd:output:0*
T0*3
_output_shapes!
:���������22@�
model/max_pooling3d/MaxPool3D	MaxPool3D!model/conv3d_1/Relu:activations:0*
T0*3
_output_shapes!
:���������
@*
ksize	
*
paddingVALID*
strides	
�
$model/conv3d_2/Conv3D/ReadVariableOpReadVariableOp-model_conv3d_2_conv3d_readvariableop_resource*+
_output_shapes
:@�*
dtype0�
model/conv3d_2/Conv3DConv3D&model/max_pooling3d/MaxPool3D:output:0,model/conv3d_2/Conv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :���������
�*
paddingSAME*
strides	
�
%model/conv3d_2/BiasAdd/ReadVariableOpReadVariableOp.model_conv3d_2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
model/conv3d_2/BiasAddBiasAddmodel/conv3d_2/Conv3D:output:0-model/conv3d_2/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :���������
�{
model/conv3d_2/ReluRelumodel/conv3d_2/BiasAdd:output:0*
T0*4
_output_shapes"
 :���������
��
$model/conv3d_3/Conv3D/ReadVariableOpReadVariableOp-model_conv3d_3_conv3d_readvariableop_resource*,
_output_shapes
:��*
dtype0�
model/conv3d_3/Conv3DConv3D!model/conv3d_2/Relu:activations:0,model/conv3d_3/Conv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :���������
�*
paddingSAME*
strides	
�
%model/conv3d_3/BiasAdd/ReadVariableOpReadVariableOp.model_conv3d_3_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
model/conv3d_3/BiasAddBiasAddmodel/conv3d_3/Conv3D:output:0-model/conv3d_3/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :���������
�{
model/conv3d_3/ReluRelumodel/conv3d_3/BiasAdd:output:0*
T0*4
_output_shapes"
 :���������
��
model/max_pooling3d_1/MaxPool3D	MaxPool3D!model/conv3d_3/Relu:activations:0*
T0*4
_output_shapes"
 :����������*
ksize	
*
paddingVALID*
strides	
�
$model/conv3d_4/Conv3D/ReadVariableOpReadVariableOp-model_conv3d_4_conv3d_readvariableop_resource*,
_output_shapes
:��*
dtype0�
model/conv3d_4/Conv3DConv3D(model/max_pooling3d_1/MaxPool3D:output:0,model/conv3d_4/Conv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :����������*
paddingSAME*
strides	
�
%model/conv3d_4/BiasAdd/ReadVariableOpReadVariableOp.model_conv3d_4_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
model/conv3d_4/BiasAddBiasAddmodel/conv3d_4/Conv3D:output:0-model/conv3d_4/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :����������{
model/conv3d_4/ReluRelumodel/conv3d_4/BiasAdd:output:0*
T0*4
_output_shapes"
 :�����������
$model/conv3d_5/Conv3D/ReadVariableOpReadVariableOp-model_conv3d_5_conv3d_readvariableop_resource*,
_output_shapes
:��*
dtype0�
model/conv3d_5/Conv3DConv3D!model/conv3d_4/Relu:activations:0,model/conv3d_5/Conv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :����������*
paddingSAME*
strides	
�
%model/conv3d_5/BiasAdd/ReadVariableOpReadVariableOp.model_conv3d_5_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
model/conv3d_5/BiasAddBiasAddmodel/conv3d_5/Conv3D:output:0-model/conv3d_5/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :����������{
model/conv3d_5/ReluRelumodel/conv3d_5/BiasAdd:output:0*
T0*4
_output_shapes"
 :�����������
model/max_pooling3d_2/MaxPool3D	MaxPool3D!model/conv3d_5/Relu:activations:0*
T0*4
_output_shapes"
 :����������*
ksize	
*
paddingVALID*
strides	
�
$model/conv3d_6/Conv3D/ReadVariableOpReadVariableOp-model_conv3d_6_conv3d_readvariableop_resource*,
_output_shapes
:��*
dtype0�
model/conv3d_6/Conv3DConv3D(model/max_pooling3d_2/MaxPool3D:output:0,model/conv3d_6/Conv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :����������*
paddingSAME*
strides	
�
%model/conv3d_6/BiasAdd/ReadVariableOpReadVariableOp.model_conv3d_6_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
model/conv3d_6/BiasAddBiasAddmodel/conv3d_6/Conv3D:output:0-model/conv3d_6/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :����������{
model/conv3d_6/ReluRelumodel/conv3d_6/BiasAdd:output:0*
T0*4
_output_shapes"
 :�����������
$model/conv3d_7/Conv3D/ReadVariableOpReadVariableOp-model_conv3d_7_conv3d_readvariableop_resource*,
_output_shapes
:��*
dtype0�
model/conv3d_7/Conv3DConv3D!model/conv3d_6/Relu:activations:0,model/conv3d_7/Conv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :����������*
paddingSAME*
strides	
�
%model/conv3d_7/BiasAdd/ReadVariableOpReadVariableOp.model_conv3d_7_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
model/conv3d_7/BiasAddBiasAddmodel/conv3d_7/Conv3D:output:0-model/conv3d_7/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :����������{
model/conv3d_7/ReluRelumodel/conv3d_7/BiasAdd:output:0*
T0*4
_output_shapes"
 :�����������
model/max_pooling3d_3/MaxPool3D	MaxPool3D!model/conv3d_7/Relu:activations:0*
T0*4
_output_shapes"
 :����������*
ksize	
*
paddingVALID*
strides	
d
model/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   �
model/flatten/ReshapeReshape(model/max_pooling3d_3/MaxPool3D:output:0model/flatten/Const:output:0*
T0*(
_output_shapes
:����������$�
!model/dense/MatMul/ReadVariableOpReadVariableOp*model_dense_matmul_readvariableop_resource*
_output_shapes
:	�$*
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
:����������
$model/conv3d/Conv3D_1/ReadVariableOpReadVariableOp+model_conv3d_conv3d_readvariableop_resource**
_output_shapes
:@*
dtype0�
model/conv3d/Conv3D_1Conv3Dinputs_0,model/conv3d/Conv3D_1/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������22@*
paddingSAME*
strides	
�
%model/conv3d/BiasAdd_1/ReadVariableOpReadVariableOp,model_conv3d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
model/conv3d/BiasAdd_1BiasAddmodel/conv3d/Conv3D_1:output:0-model/conv3d/BiasAdd_1/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������22@z
model/conv3d/Relu_1Relumodel/conv3d/BiasAdd_1:output:0*
T0*3
_output_shapes!
:���������22@�
&model/conv3d_1/Conv3D_1/ReadVariableOpReadVariableOp-model_conv3d_1_conv3d_readvariableop_resource**
_output_shapes
:@@*
dtype0�
model/conv3d_1/Conv3D_1Conv3D!model/conv3d/Relu_1:activations:0.model/conv3d_1/Conv3D_1/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������22@*
paddingSAME*
strides	
�
'model/conv3d_1/BiasAdd_1/ReadVariableOpReadVariableOp.model_conv3d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
model/conv3d_1/BiasAdd_1BiasAdd model/conv3d_1/Conv3D_1:output:0/model/conv3d_1/BiasAdd_1/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������22@~
model/conv3d_1/Relu_1Relu!model/conv3d_1/BiasAdd_1:output:0*
T0*3
_output_shapes!
:���������22@�
model/max_pooling3d/MaxPool3D_1	MaxPool3D#model/conv3d_1/Relu_1:activations:0*
T0*3
_output_shapes!
:���������
@*
ksize	
*
paddingVALID*
strides	
�
&model/conv3d_2/Conv3D_1/ReadVariableOpReadVariableOp-model_conv3d_2_conv3d_readvariableop_resource*+
_output_shapes
:@�*
dtype0�
model/conv3d_2/Conv3D_1Conv3D(model/max_pooling3d/MaxPool3D_1:output:0.model/conv3d_2/Conv3D_1/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :���������
�*
paddingSAME*
strides	
�
'model/conv3d_2/BiasAdd_1/ReadVariableOpReadVariableOp.model_conv3d_2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
model/conv3d_2/BiasAdd_1BiasAdd model/conv3d_2/Conv3D_1:output:0/model/conv3d_2/BiasAdd_1/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :���������
�
model/conv3d_2/Relu_1Relu!model/conv3d_2/BiasAdd_1:output:0*
T0*4
_output_shapes"
 :���������
��
&model/conv3d_3/Conv3D_1/ReadVariableOpReadVariableOp-model_conv3d_3_conv3d_readvariableop_resource*,
_output_shapes
:��*
dtype0�
model/conv3d_3/Conv3D_1Conv3D#model/conv3d_2/Relu_1:activations:0.model/conv3d_3/Conv3D_1/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :���������
�*
paddingSAME*
strides	
�
'model/conv3d_3/BiasAdd_1/ReadVariableOpReadVariableOp.model_conv3d_3_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
model/conv3d_3/BiasAdd_1BiasAdd model/conv3d_3/Conv3D_1:output:0/model/conv3d_3/BiasAdd_1/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :���������
�
model/conv3d_3/Relu_1Relu!model/conv3d_3/BiasAdd_1:output:0*
T0*4
_output_shapes"
 :���������
��
!model/max_pooling3d_1/MaxPool3D_1	MaxPool3D#model/conv3d_3/Relu_1:activations:0*
T0*4
_output_shapes"
 :����������*
ksize	
*
paddingVALID*
strides	
�
&model/conv3d_4/Conv3D_1/ReadVariableOpReadVariableOp-model_conv3d_4_conv3d_readvariableop_resource*,
_output_shapes
:��*
dtype0�
model/conv3d_4/Conv3D_1Conv3D*model/max_pooling3d_1/MaxPool3D_1:output:0.model/conv3d_4/Conv3D_1/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :����������*
paddingSAME*
strides	
�
'model/conv3d_4/BiasAdd_1/ReadVariableOpReadVariableOp.model_conv3d_4_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
model/conv3d_4/BiasAdd_1BiasAdd model/conv3d_4/Conv3D_1:output:0/model/conv3d_4/BiasAdd_1/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :����������
model/conv3d_4/Relu_1Relu!model/conv3d_4/BiasAdd_1:output:0*
T0*4
_output_shapes"
 :�����������
&model/conv3d_5/Conv3D_1/ReadVariableOpReadVariableOp-model_conv3d_5_conv3d_readvariableop_resource*,
_output_shapes
:��*
dtype0�
model/conv3d_5/Conv3D_1Conv3D#model/conv3d_4/Relu_1:activations:0.model/conv3d_5/Conv3D_1/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :����������*
paddingSAME*
strides	
�
'model/conv3d_5/BiasAdd_1/ReadVariableOpReadVariableOp.model_conv3d_5_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
model/conv3d_5/BiasAdd_1BiasAdd model/conv3d_5/Conv3D_1:output:0/model/conv3d_5/BiasAdd_1/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :����������
model/conv3d_5/Relu_1Relu!model/conv3d_5/BiasAdd_1:output:0*
T0*4
_output_shapes"
 :�����������
!model/max_pooling3d_2/MaxPool3D_1	MaxPool3D#model/conv3d_5/Relu_1:activations:0*
T0*4
_output_shapes"
 :����������*
ksize	
*
paddingVALID*
strides	
�
&model/conv3d_6/Conv3D_1/ReadVariableOpReadVariableOp-model_conv3d_6_conv3d_readvariableop_resource*,
_output_shapes
:��*
dtype0�
model/conv3d_6/Conv3D_1Conv3D*model/max_pooling3d_2/MaxPool3D_1:output:0.model/conv3d_6/Conv3D_1/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :����������*
paddingSAME*
strides	
�
'model/conv3d_6/BiasAdd_1/ReadVariableOpReadVariableOp.model_conv3d_6_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
model/conv3d_6/BiasAdd_1BiasAdd model/conv3d_6/Conv3D_1:output:0/model/conv3d_6/BiasAdd_1/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :����������
model/conv3d_6/Relu_1Relu!model/conv3d_6/BiasAdd_1:output:0*
T0*4
_output_shapes"
 :�����������
&model/conv3d_7/Conv3D_1/ReadVariableOpReadVariableOp-model_conv3d_7_conv3d_readvariableop_resource*,
_output_shapes
:��*
dtype0�
model/conv3d_7/Conv3D_1Conv3D#model/conv3d_6/Relu_1:activations:0.model/conv3d_7/Conv3D_1/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :����������*
paddingSAME*
strides	
�
'model/conv3d_7/BiasAdd_1/ReadVariableOpReadVariableOp.model_conv3d_7_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
model/conv3d_7/BiasAdd_1BiasAdd model/conv3d_7/Conv3D_1:output:0/model/conv3d_7/BiasAdd_1/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :����������
model/conv3d_7/Relu_1Relu!model/conv3d_7/BiasAdd_1:output:0*
T0*4
_output_shapes"
 :�����������
!model/max_pooling3d_3/MaxPool3D_1	MaxPool3D#model/conv3d_7/Relu_1:activations:0*
T0*4
_output_shapes"
 :����������*
ksize	
*
paddingVALID*
strides	
f
model/flatten/Const_1Const*
_output_shapes
:*
dtype0*
valueB"����   �
model/flatten/Reshape_1Reshape*model/max_pooling3d_3/MaxPool3D_1:output:0model/flatten/Const_1:output:0*
T0*(
_output_shapes
:����������$�
#model/dense/MatMul_1/ReadVariableOpReadVariableOp*model_dense_matmul_readvariableop_resource*
_output_shapes
:	�$*
dtype0�
model/dense/MatMul_1MatMul model/flatten/Reshape_1:output:0+model/dense/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
$model/dense/BiasAdd_1/ReadVariableOpReadVariableOp+model_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model/dense/BiasAdd_1BiasAddmodel/dense/MatMul_1:product:0,model/dense/BiasAdd_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������V
tf.compat.v1.squeeze_1/SqueezeSqueezeinputs_3*
T0*
_output_shapes
:j
tf.compat.v1.squeeze_3/SqueezeSqueezemodel/dense/BiasAdd:output:0*
T0*
_output_shapes
:T
tf.compat.v1.squeeze/SqueezeSqueezeinputs_2*
T0*
_output_shapes
:l
tf.compat.v1.squeeze_2/SqueezeSqueezemodel/dense/BiasAdd_1:output:0*
T0*
_output_shapes
:l
tf.compat.v1.squeeze_4/SqueezeSqueezemodel/dense/BiasAdd_1:output:0*
T0*
_output_shapes
:j
tf.compat.v1.squeeze_5/SqueezeSqueezemodel/dense/BiasAdd:output:0*
T0*
_output_shapes
:V
tf.compat.v1.squeeze_7/SqueezeSqueezeinputs_3*
T0*
_output_shapes
:j
tf.compat.v1.squeeze_9/SqueezeSqueezemodel/dense/BiasAdd:output:0*
T0*
_output_shapes
:V
tf.compat.v1.squeeze_6/SqueezeSqueezeinputs_2*
T0*
_output_shapes
:l
tf.compat.v1.squeeze_8/SqueezeSqueezemodel/dense/BiasAdd_1:output:0*
T0*
_output_shapes
:k
tf.compat.v1.squeeze_13/SqueezeSqueezemodel/dense/BiasAdd:output:0*
T0*
_output_shapes
:W
tf.compat.v1.squeeze_12/SqueezeSqueezeinputs_3*
T0*
_output_shapes
:m
tf.compat.v1.squeeze_11/SqueezeSqueezemodel/dense/BiasAdd_1:output:0*
T0*
_output_shapes
:W
tf.compat.v1.squeeze_10/SqueezeSqueezeinputs_2*
T0*
_output_shapes
:�
tf.math.subtract_1/SubSub'tf.compat.v1.squeeze_1/Squeeze:output:0'tf.compat.v1.squeeze_3/Squeeze:output:0*
T0*
_output_shapes
:�
tf.math.subtract/SubSub%tf.compat.v1.squeeze/Squeeze:output:0'tf.compat.v1.squeeze_2/Squeeze:output:0*
T0*
_output_shapes
:�
tf.math.subtract_2/SubSub'tf.compat.v1.squeeze_4/Squeeze:output:0'tf.compat.v1.squeeze_5/Squeeze:output:0*
T0*
_output_shapes
:�
tf.math.subtract_5/SubSub'tf.compat.v1.squeeze_7/Squeeze:output:0'tf.compat.v1.squeeze_9/Squeeze:output:0*
T0*
_output_shapes
:�
tf.math.subtract_4/SubSub'tf.compat.v1.squeeze_6/Squeeze:output:0'tf.compat.v1.squeeze_8/Squeeze:output:0*
T0*
_output_shapes
:m
tf.math.reduce_mean_8/RankRank(tf.compat.v1.squeeze_13/Squeeze:output:0*
T0*
_output_shapes
: c
!tf.math.reduce_mean_8/range/startConst*
_output_shapes
: *
dtype0*
value	B : c
!tf.math.reduce_mean_8/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
tf.math.reduce_mean_8/rangeRange*tf.math.reduce_mean_8/range/start:output:0#tf.math.reduce_mean_8/Rank:output:0*tf.math.reduce_mean_8/range/delta:output:0*#
_output_shapes
:����������
tf.math.reduce_mean_8/MeanMean(tf.compat.v1.squeeze_13/Squeeze:output:0$tf.math.reduce_mean_8/range:output:0*
T0*
_output_shapes
: m
tf.math.reduce_mean_7/RankRank(tf.compat.v1.squeeze_12/Squeeze:output:0*
T0*
_output_shapes
: c
!tf.math.reduce_mean_7/range/startConst*
_output_shapes
: *
dtype0*
value	B : c
!tf.math.reduce_mean_7/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
tf.math.reduce_mean_7/rangeRange*tf.math.reduce_mean_7/range/start:output:0#tf.math.reduce_mean_7/Rank:output:0*tf.math.reduce_mean_7/range/delta:output:0*#
_output_shapes
:����������
tf.math.reduce_mean_7/MeanMean(tf.compat.v1.squeeze_12/Squeeze:output:0$tf.math.reduce_mean_7/range:output:0*
T0*
_output_shapes
: m
tf.math.reduce_mean_6/RankRank(tf.compat.v1.squeeze_11/Squeeze:output:0*
T0*
_output_shapes
: c
!tf.math.reduce_mean_6/range/startConst*
_output_shapes
: *
dtype0*
value	B : c
!tf.math.reduce_mean_6/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
tf.math.reduce_mean_6/rangeRange*tf.math.reduce_mean_6/range/start:output:0#tf.math.reduce_mean_6/Rank:output:0*tf.math.reduce_mean_6/range/delta:output:0*#
_output_shapes
:����������
tf.math.reduce_mean_6/MeanMean(tf.compat.v1.squeeze_11/Squeeze:output:0$tf.math.reduce_mean_6/range:output:0*
T0*
_output_shapes
: m
tf.math.reduce_mean_5/RankRank(tf.compat.v1.squeeze_10/Squeeze:output:0*
T0*
_output_shapes
: c
!tf.math.reduce_mean_5/range/startConst*
_output_shapes
: *
dtype0*
value	B : c
!tf.math.reduce_mean_5/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
tf.math.reduce_mean_5/rangeRange*tf.math.reduce_mean_5/range/start:output:0#tf.math.reduce_mean_5/Rank:output:0*tf.math.reduce_mean_5/range/delta:output:0*#
_output_shapes
:����������
tf.math.reduce_mean_5/MeanMean(tf.compat.v1.squeeze_10/Squeeze:output:0$tf.math.reduce_mean_5/range:output:0*
T0*
_output_shapes
: `
tf.math.square_1/SquareSquaretf.math.subtract_1/Sub:z:0*
T0*
_output_shapes
:\
tf.math.square/SquareSquaretf.math.subtract/Sub:z:0*
T0*
_output_shapes
:`
tf.math.square_2/SquareSquaretf.math.subtract_2/Sub:z:0*
T0*
_output_shapes
:`
tf.math.square_4/SquareSquaretf.math.subtract_5/Sub:z:0*
T0*
_output_shapes
:`
tf.math.square_3/SquareSquaretf.math.subtract_4/Sub:z:0*
T0*
_output_shapes
:�
tf.math.subtract_10/SubSub(tf.compat.v1.squeeze_13/Squeeze:output:0#tf.math.reduce_mean_8/Mean:output:0*
T0*
_output_shapes
:�
tf.math.subtract_9/SubSub(tf.compat.v1.squeeze_12/Squeeze:output:0#tf.math.reduce_mean_7/Mean:output:0*
T0*
_output_shapes
:�
tf.math.subtract_7/SubSub(tf.compat.v1.squeeze_11/Squeeze:output:0#tf.math.reduce_mean_6/Mean:output:0*
T0*
_output_shapes
:�
tf.math.subtract_6/SubSub(tf.compat.v1.squeeze_10/Squeeze:output:0#tf.math.reduce_mean_5/Mean:output:0*
T0*
_output_shapes
:\
tf.math.reduce_mean/RankRanktf.math.square/Square:y:0*
T0*
_output_shapes
: a
tf.math.reduce_mean/range/startConst*
_output_shapes
: *
dtype0*
value	B : a
tf.math.reduce_mean/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
tf.math.reduce_mean/rangeRange(tf.math.reduce_mean/range/start:output:0!tf.math.reduce_mean/Rank:output:0(tf.math.reduce_mean/range/delta:output:0*#
_output_shapes
:����������
tf.math.reduce_mean/MeanMeantf.math.square/Square:y:0"tf.math.reduce_mean/range:output:0*
T0*
_output_shapes
: `
tf.math.reduce_mean_1/RankRanktf.math.square_1/Square:y:0*
T0*
_output_shapes
: c
!tf.math.reduce_mean_1/range/startConst*
_output_shapes
: *
dtype0*
value	B : c
!tf.math.reduce_mean_1/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
tf.math.reduce_mean_1/rangeRange*tf.math.reduce_mean_1/range/start:output:0#tf.math.reduce_mean_1/Rank:output:0*tf.math.reduce_mean_1/range/delta:output:0*#
_output_shapes
:����������
tf.math.reduce_mean_1/MeanMeantf.math.square_1/Square:y:0$tf.math.reduce_mean_1/range:output:0*
T0*
_output_shapes
: ^
tf.math.maximum/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *��:�
tf.math.maximum/MaximumMaximumtf.math.square_2/Square:y:0"tf.math.maximum/Maximum/y:output:0*
T0*
_output_shapes
:`
tf.math.reduce_mean_3/RankRanktf.math.square_3/Square:y:0*
T0*
_output_shapes
: c
!tf.math.reduce_mean_3/range/startConst*
_output_shapes
: *
dtype0*
value	B : c
!tf.math.reduce_mean_3/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
tf.math.reduce_mean_3/rangeRange*tf.math.reduce_mean_3/range/start:output:0#tf.math.reduce_mean_3/Rank:output:0*tf.math.reduce_mean_3/range/delta:output:0*#
_output_shapes
:����������
tf.math.reduce_mean_3/MeanMeantf.math.square_3/Square:y:0$tf.math.reduce_mean_3/range:output:0*
T0*
_output_shapes
: `
tf.math.reduce_mean_4/RankRanktf.math.square_4/Square:y:0*
T0*
_output_shapes
: c
!tf.math.reduce_mean_4/range/startConst*
_output_shapes
: *
dtype0*
value	B : c
!tf.math.reduce_mean_4/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
tf.math.reduce_mean_4/rangeRange*tf.math.reduce_mean_4/range/start:output:0#tf.math.reduce_mean_4/Rank:output:0*tf.math.reduce_mean_4/range/delta:output:0*#
_output_shapes
:����������
tf.math.reduce_mean_4/MeanMeantf.math.square_4/Square:y:0$tf.math.reduce_mean_4/range:output:0*
T0*
_output_shapes
: y
tf.math.multiply_3/MulMultf.math.subtract_9/Sub:z:0tf.math.subtract_10/Sub:z:0*
T0*
_output_shapes
:x
tf.math.multiply_1/MulMultf.math.subtract_6/Sub:z:0tf.math.subtract_7/Sub:z:0*
T0*
_output_shapes
:a
tf.math.square_8/SquareSquaretf.math.subtract_10/Sub:z:0*
T0*
_output_shapes
:`
tf.math.square_7/SquareSquaretf.math.subtract_9/Sub:z:0*
T0*
_output_shapes
:`
tf.math.square_6/SquareSquaretf.math.subtract_7/Sub:z:0*
T0*
_output_shapes
:`
tf.math.square_5/SquareSquaretf.math.subtract_6/Sub:z:0*
T0*
_output_shapes
:�
tf.__operators__.add/AddV2AddV2!tf.math.reduce_mean/Mean:output:0#tf.math.reduce_mean_1/Mean:output:0*
T0*
_output_shapes
: ]
tf.math.subtract_3/Sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *��:�
tf.math.subtract_3/SubSubtf.math.maximum/Maximum:z:0!tf.math.subtract_3/Sub/y:output:0*
T0*
_output_shapes
:�
tf.__operators__.add_2/AddV2AddV2#tf.math.reduce_mean_3/Mean:output:0#tf.math.reduce_mean_4/Mean:output:0*
T0*
_output_shapes
: ^
tf.math.reduce_sum_3/RankRanktf.math.multiply_3/Mul:z:0*
T0*
_output_shapes
: b
 tf.math.reduce_sum_3/range/startConst*
_output_shapes
: *
dtype0*
value	B : b
 tf.math.reduce_sum_3/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
tf.math.reduce_sum_3/rangeRange)tf.math.reduce_sum_3/range/start:output:0"tf.math.reduce_sum_3/Rank:output:0)tf.math.reduce_sum_3/range/delta:output:0*#
_output_shapes
:����������
tf.math.reduce_sum_3/SumSumtf.math.multiply_3/Mul:z:0#tf.math.reduce_sum_3/range:output:0*
T0*
_output_shapes
: \
tf.math.reduce_sum/RankRanktf.math.multiply_1/Mul:z:0*
T0*
_output_shapes
: `
tf.math.reduce_sum/range/startConst*
_output_shapes
: *
dtype0*
value	B : `
tf.math.reduce_sum/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
tf.math.reduce_sum/rangeRange'tf.math.reduce_sum/range/start:output:0 tf.math.reduce_sum/Rank:output:0'tf.math.reduce_sum/range/delta:output:0*#
_output_shapes
:���������}
tf.math.reduce_sum/SumSumtf.math.multiply_1/Mul:z:0!tf.math.reduce_sum/range:output:0*
T0*
_output_shapes
: _
tf.math.reduce_sum_4/RankRanktf.math.square_7/Square:y:0*
T0*
_output_shapes
: b
 tf.math.reduce_sum_4/range/startConst*
_output_shapes
: *
dtype0*
value	B : b
 tf.math.reduce_sum_4/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
tf.math.reduce_sum_4/rangeRange)tf.math.reduce_sum_4/range/start:output:0"tf.math.reduce_sum_4/Rank:output:0)tf.math.reduce_sum_4/range/delta:output:0*#
_output_shapes
:����������
tf.math.reduce_sum_4/SumSumtf.math.square_7/Square:y:0#tf.math.reduce_sum_4/range:output:0*
T0*
_output_shapes
: _
tf.math.reduce_sum_5/RankRanktf.math.square_8/Square:y:0*
T0*
_output_shapes
: b
 tf.math.reduce_sum_5/range/startConst*
_output_shapes
: *
dtype0*
value	B : b
 tf.math.reduce_sum_5/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
tf.math.reduce_sum_5/rangeRange)tf.math.reduce_sum_5/range/start:output:0"tf.math.reduce_sum_5/Rank:output:0)tf.math.reduce_sum_5/range/delta:output:0*#
_output_shapes
:����������
tf.math.reduce_sum_5/SumSumtf.math.square_8/Square:y:0#tf.math.reduce_sum_5/range:output:0*
T0*
_output_shapes
: _
tf.math.reduce_sum_1/RankRanktf.math.square_5/Square:y:0*
T0*
_output_shapes
: b
 tf.math.reduce_sum_1/range/startConst*
_output_shapes
: *
dtype0*
value	B : b
 tf.math.reduce_sum_1/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
tf.math.reduce_sum_1/rangeRange)tf.math.reduce_sum_1/range/start:output:0"tf.math.reduce_sum_1/Rank:output:0)tf.math.reduce_sum_1/range/delta:output:0*#
_output_shapes
:����������
tf.math.reduce_sum_1/SumSumtf.math.square_5/Square:y:0#tf.math.reduce_sum_1/range:output:0*
T0*
_output_shapes
: _
tf.math.reduce_sum_2/RankRanktf.math.square_6/Square:y:0*
T0*
_output_shapes
: b
 tf.math.reduce_sum_2/range/startConst*
_output_shapes
: *
dtype0*
value	B : b
 tf.math.reduce_sum_2/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
tf.math.reduce_sum_2/rangeRange)tf.math.reduce_sum_2/range/start:output:0"tf.math.reduce_sum_2/Rank:output:0)tf.math.reduce_sum_2/range/delta:output:0*#
_output_shapes
:����������
tf.math.reduce_sum_2/SumSumtf.math.square_6/Square:y:0#tf.math.reduce_sum_2/range:output:0*
T0*
_output_shapes
: _
tf.math.reduce_mean_2/RankRanktf.math.subtract_3/Sub:z:0*
T0*
_output_shapes
: c
!tf.math.reduce_mean_2/range/startConst*
_output_shapes
: *
dtype0*
value	B : c
!tf.math.reduce_mean_2/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
tf.math.reduce_mean_2/rangeRange*tf.math.reduce_mean_2/range/start:output:0#tf.math.reduce_mean_2/Rank:output:0*tf.math.reduce_mean_2/range/delta:output:0*#
_output_shapes
:����������
tf.math.reduce_mean_2/MeanMeantf.math.subtract_3/Sub:z:0$tf.math.reduce_mean_2/range:output:0*
T0*
_output_shapes
: Q
add_metric/RankConst*
_output_shapes
: *
dtype0*
value	B : X
add_metric/range/startConst*
_output_shapes
: *
dtype0*
value	B : X
add_metric/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
add_metric/rangeRangeadd_metric/range/start:output:0add_metric/Rank:output:0add_metric/range/delta:output:0*
_output_shapes
: s
add_metric/SumSum tf.__operators__.add_2/AddV2:z:0add_metric/range:output:0*
T0*
_output_shapes
: �
add_metric/AssignAddVariableOpAssignAddVariableOp'add_metric_assignaddvariableop_resourceadd_metric/Sum:output:0*
_output_shapes
 *
dtype0Q
add_metric/SizeConst*
_output_shapes
: *
dtype0*
value	B :a
add_metric/CastCastadd_metric/Size:output:0*

DstT0*

SrcT0*
_output_shapes
: �
 add_metric/AssignAddVariableOp_1AssignAddVariableOp)add_metric_assignaddvariableop_1_resourceadd_metric/Cast:y:0^add_metric/AssignAddVariableOp*
_output_shapes
 *
dtype0�
$add_metric/div_no_nan/ReadVariableOpReadVariableOp'add_metric_assignaddvariableop_resource^add_metric/AssignAddVariableOp!^add_metric/AssignAddVariableOp_1*
_output_shapes
: *
dtype0�
&add_metric/div_no_nan/ReadVariableOp_1ReadVariableOp)add_metric_assignaddvariableop_1_resource!^add_metric/AssignAddVariableOp_1*
_output_shapes
: *
dtype0�
add_metric/div_no_nanDivNoNan,add_metric/div_no_nan/ReadVariableOp:value:0.add_metric/div_no_nan/ReadVariableOp_1:value:0*
T0*
_output_shapes
: [
add_metric/IdentityIdentityadd_metric/div_no_nan:z:0*
T0*
_output_shapes
: �
tf.math.multiply_4/MulMul!tf.math.reduce_sum_4/Sum:output:0!tf.math.reduce_sum_5/Sum:output:0*
T0*
_output_shapes
: �
tf.math.multiply_2/MulMul!tf.math.reduce_sum_1/Sum:output:0!tf.math.reduce_sum_2/Sum:output:0*
T0*
_output_shapes
: j
tf.math.multiply/MulMulunknown#tf.math.reduce_mean_2/Mean:output:0*
T0*
_output_shapes
: X
tf.math.sqrt_1/SqrtSqrttf.math.multiply_4/Mul:z:0*
T0*
_output_shapes
: V
tf.math.sqrt/SqrtSqrttf.math.multiply_2/Mul:z:0*
T0*
_output_shapes
: �
tf.__operators__.add_1/AddV2AddV2tf.__operators__.add/AddV2:z:0tf.math.multiply/Mul:z:0*
T0*
_output_shapes
: ]
tf.__operators__.add_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *��'7�
tf.__operators__.add_4/AddV2AddV2tf.math.sqrt_1/Sqrt:y:0!tf.__operators__.add_4/y:output:0*
T0*
_output_shapes
: ]
tf.__operators__.add_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *��'7�
tf.__operators__.add_3/AddV2AddV2tf.math.sqrt/Sqrt:y:0!tf.__operators__.add_3/y:output:0*
T0*
_output_shapes
: �
tf.math.truediv_1/truedivRealDiv!tf.math.reduce_sum_3/Sum:output:0 tf.__operators__.add_4/AddV2:z:0*
T0*
_output_shapes
: �
tf.math.truediv/truedivRealDivtf.math.reduce_sum/Sum:output:0 tf.__operators__.add_3/AddV2:z:0*
T0*
_output_shapes
: i
tf.math.subtract_11/SubSub	unknown_0tf.math.truediv_1/truediv:z:0*
T0*
_output_shapes
: f
tf.math.subtract_8/SubSub	unknown_1tf.math.truediv/truediv:z:0*
T0*
_output_shapes
: V
tf.math.pow/Pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?o
tf.math.pow/PowPowtf.math.subtract_8/Sub:z:0tf.math.pow/Pow/y:output:0*
T0*
_output_shapes
: X
tf.math.pow_1/Pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?t
tf.math.pow_1/PowPowtf.math.subtract_11/Sub:z:0tf.math.pow_1/Pow/y:output:0*
T0*
_output_shapes
: r
tf.__operators__.add_5/AddV2AddV2tf.math.pow/Pow:z:0tf.math.pow_1/Pow:z:0*
T0*
_output_shapes
: S
add_metric_1/RankConst*
_output_shapes
: *
dtype0*
value	B : Z
add_metric_1/range/startConst*
_output_shapes
: *
dtype0*
value	B : Z
add_metric_1/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
add_metric_1/rangeRange!add_metric_1/range/start:output:0add_metric_1/Rank:output:0!add_metric_1/range/delta:output:0*
_output_shapes
: w
add_metric_1/SumSum tf.__operators__.add_5/AddV2:z:0add_metric_1/range:output:0*
T0*
_output_shapes
: �
 add_metric_1/AssignAddVariableOpAssignAddVariableOp)add_metric_1_assignaddvariableop_resourceadd_metric_1/Sum:output:0*
_output_shapes
 *
dtype0S
add_metric_1/SizeConst*
_output_shapes
: *
dtype0*
value	B :e
add_metric_1/CastCastadd_metric_1/Size:output:0*

DstT0*

SrcT0*
_output_shapes
: �
"add_metric_1/AssignAddVariableOp_1AssignAddVariableOp+add_metric_1_assignaddvariableop_1_resourceadd_metric_1/Cast:y:0!^add_metric_1/AssignAddVariableOp*
_output_shapes
 *
dtype0�
&add_metric_1/div_no_nan/ReadVariableOpReadVariableOp)add_metric_1_assignaddvariableop_resource!^add_metric_1/AssignAddVariableOp#^add_metric_1/AssignAddVariableOp_1*
_output_shapes
: *
dtype0�
(add_metric_1/div_no_nan/ReadVariableOp_1ReadVariableOp+add_metric_1_assignaddvariableop_1_resource#^add_metric_1/AssignAddVariableOp_1*
_output_shapes
: *
dtype0�
add_metric_1/div_no_nanDivNoNan.add_metric_1/div_no_nan/ReadVariableOp:value:00add_metric_1/div_no_nan/ReadVariableOp_1:value:0*
T0*
_output_shapes
: _
add_metric_1/IdentityIdentityadd_metric_1/div_no_nan:z:0*
T0*
_output_shapes
: �
/conv3d/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+model_conv3d_conv3d_readvariableop_resource**
_output_shapes
:@*
dtype0�
 conv3d/kernel/Regularizer/SquareSquare7conv3d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0**
_output_shapes
:@|
conv3d/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
conv3d/kernel/Regularizer/SumSum$conv3d/kernel/Regularizer/Square:y:0(conv3d/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: d
conv3d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    �
conv3d/kernel/Regularizer/mulMul(conv3d/kernel/Regularizer/mul/x:output:0&conv3d/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1conv3d_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp-model_conv3d_1_conv3d_readvariableop_resource**
_output_shapes
:@@*
dtype0�
"conv3d_1/kernel/Regularizer/SquareSquare9conv3d_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0**
_output_shapes
:@@~
!conv3d_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
conv3d_1/kernel/Regularizer/SumSum&conv3d_1/kernel/Regularizer/Square:y:0*conv3d_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!conv3d_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    �
conv3d_1/kernel/Regularizer/mulMul*conv3d_1/kernel/Regularizer/mul/x:output:0(conv3d_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1conv3d_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp-model_conv3d_2_conv3d_readvariableop_resource*+
_output_shapes
:@�*
dtype0�
"conv3d_2/kernel/Regularizer/SquareSquare9conv3d_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*+
_output_shapes
:@�~
!conv3d_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
conv3d_2/kernel/Regularizer/SumSum&conv3d_2/kernel/Regularizer/Square:y:0*conv3d_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!conv3d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    �
conv3d_2/kernel/Regularizer/mulMul*conv3d_2/kernel/Regularizer/mul/x:output:0(conv3d_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1conv3d_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOp-model_conv3d_3_conv3d_readvariableop_resource*,
_output_shapes
:��*
dtype0�
"conv3d_3/kernel/Regularizer/SquareSquare9conv3d_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*,
_output_shapes
:��~
!conv3d_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
conv3d_3/kernel/Regularizer/SumSum&conv3d_3/kernel/Regularizer/Square:y:0*conv3d_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!conv3d_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    �
conv3d_3/kernel/Regularizer/mulMul*conv3d_3/kernel/Regularizer/mul/x:output:0(conv3d_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1conv3d_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOp-model_conv3d_4_conv3d_readvariableop_resource*,
_output_shapes
:��*
dtype0�
"conv3d_4/kernel/Regularizer/SquareSquare9conv3d_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*,
_output_shapes
:��~
!conv3d_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
conv3d_4/kernel/Regularizer/SumSum&conv3d_4/kernel/Regularizer/Square:y:0*conv3d_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!conv3d_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    �
conv3d_4/kernel/Regularizer/mulMul*conv3d_4/kernel/Regularizer/mul/x:output:0(conv3d_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1conv3d_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOp-model_conv3d_5_conv3d_readvariableop_resource*,
_output_shapes
:��*
dtype0�
"conv3d_5/kernel/Regularizer/SquareSquare9conv3d_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*,
_output_shapes
:��~
!conv3d_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
conv3d_5/kernel/Regularizer/SumSum&conv3d_5/kernel/Regularizer/Square:y:0*conv3d_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!conv3d_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    �
conv3d_5/kernel/Regularizer/mulMul*conv3d_5/kernel/Regularizer/mul/x:output:0(conv3d_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1conv3d_6/kernel/Regularizer/Square/ReadVariableOpReadVariableOp-model_conv3d_6_conv3d_readvariableop_resource*,
_output_shapes
:��*
dtype0�
"conv3d_6/kernel/Regularizer/SquareSquare9conv3d_6/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*,
_output_shapes
:��~
!conv3d_6/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
conv3d_6/kernel/Regularizer/SumSum&conv3d_6/kernel/Regularizer/Square:y:0*conv3d_6/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!conv3d_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    �
conv3d_6/kernel/Regularizer/mulMul*conv3d_6/kernel/Regularizer/mul/x:output:0(conv3d_6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1conv3d_7/kernel/Regularizer/Square/ReadVariableOpReadVariableOp-model_conv3d_7_conv3d_readvariableop_resource*,
_output_shapes
:��*
dtype0�
"conv3d_7/kernel/Regularizer/SquareSquare9conv3d_7/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*,
_output_shapes
:��~
!conv3d_7/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
conv3d_7/kernel/Regularizer/SumSum&conv3d_7/kernel/Regularizer/Square:y:0*conv3d_7/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!conv3d_7/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    �
conv3d_7/kernel/Regularizer/mulMul*conv3d_7/kernel/Regularizer/mul/x:output:0(conv3d_7/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOp*model_dense_matmul_readvariableop_resource*
_output_shapes
:	�$*
dtype0�
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	�$o
dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
dense/kernel/Regularizer/SumSum#dense/kernel/Regularizer/Square:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: c
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    �
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: m
IdentityIdentitymodel/dense/BiasAdd_1:output:0^NoOp*
T0*'
_output_shapes
:���������m

Identity_1Identitymodel/dense/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������`

Identity_2Identity tf.__operators__.add_1/AddV2:z:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp^add_metric/AssignAddVariableOp!^add_metric/AssignAddVariableOp_1%^add_metric/div_no_nan/ReadVariableOp'^add_metric/div_no_nan/ReadVariableOp_1!^add_metric_1/AssignAddVariableOp#^add_metric_1/AssignAddVariableOp_1'^add_metric_1/div_no_nan/ReadVariableOp)^add_metric_1/div_no_nan/ReadVariableOp_10^conv3d/kernel/Regularizer/Square/ReadVariableOp2^conv3d_1/kernel/Regularizer/Square/ReadVariableOp2^conv3d_2/kernel/Regularizer/Square/ReadVariableOp2^conv3d_3/kernel/Regularizer/Square/ReadVariableOp2^conv3d_4/kernel/Regularizer/Square/ReadVariableOp2^conv3d_5/kernel/Regularizer/Square/ReadVariableOp2^conv3d_6/kernel/Regularizer/Square/ReadVariableOp2^conv3d_7/kernel/Regularizer/Square/ReadVariableOp/^dense/kernel/Regularizer/Square/ReadVariableOp$^model/conv3d/BiasAdd/ReadVariableOp&^model/conv3d/BiasAdd_1/ReadVariableOp#^model/conv3d/Conv3D/ReadVariableOp%^model/conv3d/Conv3D_1/ReadVariableOp&^model/conv3d_1/BiasAdd/ReadVariableOp(^model/conv3d_1/BiasAdd_1/ReadVariableOp%^model/conv3d_1/Conv3D/ReadVariableOp'^model/conv3d_1/Conv3D_1/ReadVariableOp&^model/conv3d_2/BiasAdd/ReadVariableOp(^model/conv3d_2/BiasAdd_1/ReadVariableOp%^model/conv3d_2/Conv3D/ReadVariableOp'^model/conv3d_2/Conv3D_1/ReadVariableOp&^model/conv3d_3/BiasAdd/ReadVariableOp(^model/conv3d_3/BiasAdd_1/ReadVariableOp%^model/conv3d_3/Conv3D/ReadVariableOp'^model/conv3d_3/Conv3D_1/ReadVariableOp&^model/conv3d_4/BiasAdd/ReadVariableOp(^model/conv3d_4/BiasAdd_1/ReadVariableOp%^model/conv3d_4/Conv3D/ReadVariableOp'^model/conv3d_4/Conv3D_1/ReadVariableOp&^model/conv3d_5/BiasAdd/ReadVariableOp(^model/conv3d_5/BiasAdd_1/ReadVariableOp%^model/conv3d_5/Conv3D/ReadVariableOp'^model/conv3d_5/Conv3D_1/ReadVariableOp&^model/conv3d_6/BiasAdd/ReadVariableOp(^model/conv3d_6/BiasAdd_1/ReadVariableOp%^model/conv3d_6/Conv3D/ReadVariableOp'^model/conv3d_6/Conv3D_1/ReadVariableOp&^model/conv3d_7/BiasAdd/ReadVariableOp(^model/conv3d_7/BiasAdd_1/ReadVariableOp%^model/conv3d_7/Conv3D/ReadVariableOp'^model/conv3d_7/Conv3D_1/ReadVariableOp#^model/dense/BiasAdd/ReadVariableOp%^model/dense/BiasAdd_1/ReadVariableOp"^model/dense/MatMul/ReadVariableOp$^model/dense/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������22:���������22:���������:���������: : : : : : : : : : : : : : : : : : : : : : : : : 2@
add_metric/AssignAddVariableOpadd_metric/AssignAddVariableOp2D
 add_metric/AssignAddVariableOp_1 add_metric/AssignAddVariableOp_12L
$add_metric/div_no_nan/ReadVariableOp$add_metric/div_no_nan/ReadVariableOp2P
&add_metric/div_no_nan/ReadVariableOp_1&add_metric/div_no_nan/ReadVariableOp_12D
 add_metric_1/AssignAddVariableOp add_metric_1/AssignAddVariableOp2H
"add_metric_1/AssignAddVariableOp_1"add_metric_1/AssignAddVariableOp_12P
&add_metric_1/div_no_nan/ReadVariableOp&add_metric_1/div_no_nan/ReadVariableOp2T
(add_metric_1/div_no_nan/ReadVariableOp_1(add_metric_1/div_no_nan/ReadVariableOp_12b
/conv3d/kernel/Regularizer/Square/ReadVariableOp/conv3d/kernel/Regularizer/Square/ReadVariableOp2f
1conv3d_1/kernel/Regularizer/Square/ReadVariableOp1conv3d_1/kernel/Regularizer/Square/ReadVariableOp2f
1conv3d_2/kernel/Regularizer/Square/ReadVariableOp1conv3d_2/kernel/Regularizer/Square/ReadVariableOp2f
1conv3d_3/kernel/Regularizer/Square/ReadVariableOp1conv3d_3/kernel/Regularizer/Square/ReadVariableOp2f
1conv3d_4/kernel/Regularizer/Square/ReadVariableOp1conv3d_4/kernel/Regularizer/Square/ReadVariableOp2f
1conv3d_5/kernel/Regularizer/Square/ReadVariableOp1conv3d_5/kernel/Regularizer/Square/ReadVariableOp2f
1conv3d_6/kernel/Regularizer/Square/ReadVariableOp1conv3d_6/kernel/Regularizer/Square/ReadVariableOp2f
1conv3d_7/kernel/Regularizer/Square/ReadVariableOp1conv3d_7/kernel/Regularizer/Square/ReadVariableOp2`
.dense/kernel/Regularizer/Square/ReadVariableOp.dense/kernel/Regularizer/Square/ReadVariableOp2J
#model/conv3d/BiasAdd/ReadVariableOp#model/conv3d/BiasAdd/ReadVariableOp2N
%model/conv3d/BiasAdd_1/ReadVariableOp%model/conv3d/BiasAdd_1/ReadVariableOp2H
"model/conv3d/Conv3D/ReadVariableOp"model/conv3d/Conv3D/ReadVariableOp2L
$model/conv3d/Conv3D_1/ReadVariableOp$model/conv3d/Conv3D_1/ReadVariableOp2N
%model/conv3d_1/BiasAdd/ReadVariableOp%model/conv3d_1/BiasAdd/ReadVariableOp2R
'model/conv3d_1/BiasAdd_1/ReadVariableOp'model/conv3d_1/BiasAdd_1/ReadVariableOp2L
$model/conv3d_1/Conv3D/ReadVariableOp$model/conv3d_1/Conv3D/ReadVariableOp2P
&model/conv3d_1/Conv3D_1/ReadVariableOp&model/conv3d_1/Conv3D_1/ReadVariableOp2N
%model/conv3d_2/BiasAdd/ReadVariableOp%model/conv3d_2/BiasAdd/ReadVariableOp2R
'model/conv3d_2/BiasAdd_1/ReadVariableOp'model/conv3d_2/BiasAdd_1/ReadVariableOp2L
$model/conv3d_2/Conv3D/ReadVariableOp$model/conv3d_2/Conv3D/ReadVariableOp2P
&model/conv3d_2/Conv3D_1/ReadVariableOp&model/conv3d_2/Conv3D_1/ReadVariableOp2N
%model/conv3d_3/BiasAdd/ReadVariableOp%model/conv3d_3/BiasAdd/ReadVariableOp2R
'model/conv3d_3/BiasAdd_1/ReadVariableOp'model/conv3d_3/BiasAdd_1/ReadVariableOp2L
$model/conv3d_3/Conv3D/ReadVariableOp$model/conv3d_3/Conv3D/ReadVariableOp2P
&model/conv3d_3/Conv3D_1/ReadVariableOp&model/conv3d_3/Conv3D_1/ReadVariableOp2N
%model/conv3d_4/BiasAdd/ReadVariableOp%model/conv3d_4/BiasAdd/ReadVariableOp2R
'model/conv3d_4/BiasAdd_1/ReadVariableOp'model/conv3d_4/BiasAdd_1/ReadVariableOp2L
$model/conv3d_4/Conv3D/ReadVariableOp$model/conv3d_4/Conv3D/ReadVariableOp2P
&model/conv3d_4/Conv3D_1/ReadVariableOp&model/conv3d_4/Conv3D_1/ReadVariableOp2N
%model/conv3d_5/BiasAdd/ReadVariableOp%model/conv3d_5/BiasAdd/ReadVariableOp2R
'model/conv3d_5/BiasAdd_1/ReadVariableOp'model/conv3d_5/BiasAdd_1/ReadVariableOp2L
$model/conv3d_5/Conv3D/ReadVariableOp$model/conv3d_5/Conv3D/ReadVariableOp2P
&model/conv3d_5/Conv3D_1/ReadVariableOp&model/conv3d_5/Conv3D_1/ReadVariableOp2N
%model/conv3d_6/BiasAdd/ReadVariableOp%model/conv3d_6/BiasAdd/ReadVariableOp2R
'model/conv3d_6/BiasAdd_1/ReadVariableOp'model/conv3d_6/BiasAdd_1/ReadVariableOp2L
$model/conv3d_6/Conv3D/ReadVariableOp$model/conv3d_6/Conv3D/ReadVariableOp2P
&model/conv3d_6/Conv3D_1/ReadVariableOp&model/conv3d_6/Conv3D_1/ReadVariableOp2N
%model/conv3d_7/BiasAdd/ReadVariableOp%model/conv3d_7/BiasAdd/ReadVariableOp2R
'model/conv3d_7/BiasAdd_1/ReadVariableOp'model/conv3d_7/BiasAdd_1/ReadVariableOp2L
$model/conv3d_7/Conv3D/ReadVariableOp$model/conv3d_7/Conv3D/ReadVariableOp2P
&model/conv3d_7/Conv3D_1/ReadVariableOp&model/conv3d_7/Conv3D_1/ReadVariableOp2H
"model/dense/BiasAdd/ReadVariableOp"model/dense/BiasAdd/ReadVariableOp2L
$model/dense/BiasAdd_1/ReadVariableOp$model/dense/BiasAdd_1/ReadVariableOp2F
!model/dense/MatMul/ReadVariableOp!model/dense/MatMul/ReadVariableOp2J
#model/dense/MatMul_1/ReadVariableOp#model/dense/MatMul_1/ReadVariableOp:] Y
3
_output_shapes!
:���������22
"
_user_specified_name
inputs/0:]Y
3
_output_shapes!
:���������22
"
_user_specified_name
inputs/1:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs/2:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs/3:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
�
@__inference_model_layer_call_and_return_conditional_losses_13926
input_1*
conv3d_13821:@
conv3d_13823:@,
conv3d_1_13826:@@
conv3d_1_13828:@-
conv3d_2_13832:@�
conv3d_2_13834:	�.
conv3d_3_13837:��
conv3d_3_13839:	�.
conv3d_4_13843:��
conv3d_4_13845:	�.
conv3d_5_13848:��
conv3d_5_13850:	�.
conv3d_6_13854:��
conv3d_6_13856:	�.
conv3d_7_13859:��
conv3d_7_13861:	�
dense_13866:	�$
dense_13868:
identity��conv3d/StatefulPartitionedCall�/conv3d/kernel/Regularizer/Square/ReadVariableOp� conv3d_1/StatefulPartitionedCall�1conv3d_1/kernel/Regularizer/Square/ReadVariableOp� conv3d_2/StatefulPartitionedCall�1conv3d_2/kernel/Regularizer/Square/ReadVariableOp� conv3d_3/StatefulPartitionedCall�1conv3d_3/kernel/Regularizer/Square/ReadVariableOp� conv3d_4/StatefulPartitionedCall�1conv3d_4/kernel/Regularizer/Square/ReadVariableOp� conv3d_5/StatefulPartitionedCall�1conv3d_5/kernel/Regularizer/Square/ReadVariableOp� conv3d_6/StatefulPartitionedCall�1conv3d_6/kernel/Regularizer/Square/ReadVariableOp� conv3d_7/StatefulPartitionedCall�1conv3d_7/kernel/Regularizer/Square/ReadVariableOp�dense/StatefulPartitionedCall�.dense/kernel/Regularizer/Square/ReadVariableOp�
conv3d/StatefulPartitionedCallStatefulPartitionedCallinput_1conv3d_13821conv3d_13823*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������22@*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *J
fERC
A__inference_conv3d_layer_call_and_return_conditional_losses_13196�
 conv3d_1/StatefulPartitionedCallStatefulPartitionedCall'conv3d/StatefulPartitionedCall:output:0conv3d_1_13826conv3d_1_13828*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������22@*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *L
fGRE
C__inference_conv3d_1_layer_call_and_return_conditional_losses_13219�
max_pooling3d/PartitionedCallPartitionedCall)conv3d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������
@* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8� *Q
fLRJ
H__inference_max_pooling3d_layer_call_and_return_conditional_losses_13133�
 conv3d_2/StatefulPartitionedCallStatefulPartitionedCall&max_pooling3d/PartitionedCall:output:0conv3d_2_13832conv3d_2_13834*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :���������
�*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *L
fGRE
C__inference_conv3d_2_layer_call_and_return_conditional_losses_13243�
 conv3d_3/StatefulPartitionedCallStatefulPartitionedCall)conv3d_2/StatefulPartitionedCall:output:0conv3d_3_13837conv3d_3_13839*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :���������
�*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *L
fGRE
C__inference_conv3d_3_layer_call_and_return_conditional_losses_13266�
max_pooling3d_1/PartitionedCallPartitionedCall)conv3d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :����������* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8� *S
fNRL
J__inference_max_pooling3d_1_layer_call_and_return_conditional_losses_13145�
 conv3d_4/StatefulPartitionedCallStatefulPartitionedCall(max_pooling3d_1/PartitionedCall:output:0conv3d_4_13843conv3d_4_13845*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :����������*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *L
fGRE
C__inference_conv3d_4_layer_call_and_return_conditional_losses_13290�
 conv3d_5/StatefulPartitionedCallStatefulPartitionedCall)conv3d_4/StatefulPartitionedCall:output:0conv3d_5_13848conv3d_5_13850*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :����������*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *L
fGRE
C__inference_conv3d_5_layer_call_and_return_conditional_losses_13313�
max_pooling3d_2/PartitionedCallPartitionedCall)conv3d_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :����������* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8� *S
fNRL
J__inference_max_pooling3d_2_layer_call_and_return_conditional_losses_13157�
 conv3d_6/StatefulPartitionedCallStatefulPartitionedCall(max_pooling3d_2/PartitionedCall:output:0conv3d_6_13854conv3d_6_13856*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :����������*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *L
fGRE
C__inference_conv3d_6_layer_call_and_return_conditional_losses_13337�
 conv3d_7/StatefulPartitionedCallStatefulPartitionedCall)conv3d_6/StatefulPartitionedCall:output:0conv3d_7_13859conv3d_7_13861*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :����������*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *L
fGRE
C__inference_conv3d_7_layer_call_and_return_conditional_losses_13360�
max_pooling3d_3/PartitionedCallPartitionedCall)conv3d_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :����������* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8� *S
fNRL
J__inference_max_pooling3d_3_layer_call_and_return_conditional_losses_13169�
flatten/PartitionedCallPartitionedCall(max_pooling3d_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������$* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8� *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_13373�
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_13866dense_13868*
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
@__inference_dense_layer_call_and_return_conditional_losses_13391�
/conv3d/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv3d_13821**
_output_shapes
:@*
dtype0�
 conv3d/kernel/Regularizer/SquareSquare7conv3d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0**
_output_shapes
:@|
conv3d/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
conv3d/kernel/Regularizer/SumSum$conv3d/kernel/Regularizer/Square:y:0(conv3d/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: d
conv3d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    �
conv3d/kernel/Regularizer/mulMul(conv3d/kernel/Regularizer/mul/x:output:0&conv3d/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1conv3d_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv3d_1_13826**
_output_shapes
:@@*
dtype0�
"conv3d_1/kernel/Regularizer/SquareSquare9conv3d_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0**
_output_shapes
:@@~
!conv3d_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
conv3d_1/kernel/Regularizer/SumSum&conv3d_1/kernel/Regularizer/Square:y:0*conv3d_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!conv3d_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    �
conv3d_1/kernel/Regularizer/mulMul*conv3d_1/kernel/Regularizer/mul/x:output:0(conv3d_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1conv3d_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv3d_2_13832*+
_output_shapes
:@�*
dtype0�
"conv3d_2/kernel/Regularizer/SquareSquare9conv3d_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*+
_output_shapes
:@�~
!conv3d_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
conv3d_2/kernel/Regularizer/SumSum&conv3d_2/kernel/Regularizer/Square:y:0*conv3d_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!conv3d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    �
conv3d_2/kernel/Regularizer/mulMul*conv3d_2/kernel/Regularizer/mul/x:output:0(conv3d_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1conv3d_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv3d_3_13837*,
_output_shapes
:��*
dtype0�
"conv3d_3/kernel/Regularizer/SquareSquare9conv3d_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*,
_output_shapes
:��~
!conv3d_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
conv3d_3/kernel/Regularizer/SumSum&conv3d_3/kernel/Regularizer/Square:y:0*conv3d_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!conv3d_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    �
conv3d_3/kernel/Regularizer/mulMul*conv3d_3/kernel/Regularizer/mul/x:output:0(conv3d_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1conv3d_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv3d_4_13843*,
_output_shapes
:��*
dtype0�
"conv3d_4/kernel/Regularizer/SquareSquare9conv3d_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*,
_output_shapes
:��~
!conv3d_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
conv3d_4/kernel/Regularizer/SumSum&conv3d_4/kernel/Regularizer/Square:y:0*conv3d_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!conv3d_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    �
conv3d_4/kernel/Regularizer/mulMul*conv3d_4/kernel/Regularizer/mul/x:output:0(conv3d_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1conv3d_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv3d_5_13848*,
_output_shapes
:��*
dtype0�
"conv3d_5/kernel/Regularizer/SquareSquare9conv3d_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*,
_output_shapes
:��~
!conv3d_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
conv3d_5/kernel/Regularizer/SumSum&conv3d_5/kernel/Regularizer/Square:y:0*conv3d_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!conv3d_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    �
conv3d_5/kernel/Regularizer/mulMul*conv3d_5/kernel/Regularizer/mul/x:output:0(conv3d_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1conv3d_6/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv3d_6_13854*,
_output_shapes
:��*
dtype0�
"conv3d_6/kernel/Regularizer/SquareSquare9conv3d_6/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*,
_output_shapes
:��~
!conv3d_6/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
conv3d_6/kernel/Regularizer/SumSum&conv3d_6/kernel/Regularizer/Square:y:0*conv3d_6/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!conv3d_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    �
conv3d_6/kernel/Regularizer/mulMul*conv3d_6/kernel/Regularizer/mul/x:output:0(conv3d_6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1conv3d_7/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv3d_7_13859*,
_output_shapes
:��*
dtype0�
"conv3d_7/kernel/Regularizer/SquareSquare9conv3d_7/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*,
_output_shapes
:��~
!conv3d_7/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
conv3d_7/kernel/Regularizer/SumSum&conv3d_7/kernel/Regularizer/Square:y:0*conv3d_7/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!conv3d_7/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    �
conv3d_7/kernel/Regularizer/mulMul*conv3d_7/kernel/Regularizer/mul/x:output:0(conv3d_7/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: {
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_13866*
_output_shapes
:	�$*
dtype0�
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	�$o
dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
dense/kernel/Regularizer/SumSum#dense/kernel/Regularizer/Square:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: c
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    �
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: u
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^conv3d/StatefulPartitionedCall0^conv3d/kernel/Regularizer/Square/ReadVariableOp!^conv3d_1/StatefulPartitionedCall2^conv3d_1/kernel/Regularizer/Square/ReadVariableOp!^conv3d_2/StatefulPartitionedCall2^conv3d_2/kernel/Regularizer/Square/ReadVariableOp!^conv3d_3/StatefulPartitionedCall2^conv3d_3/kernel/Regularizer/Square/ReadVariableOp!^conv3d_4/StatefulPartitionedCall2^conv3d_4/kernel/Regularizer/Square/ReadVariableOp!^conv3d_5/StatefulPartitionedCall2^conv3d_5/kernel/Regularizer/Square/ReadVariableOp!^conv3d_6/StatefulPartitionedCall2^conv3d_6/kernel/Regularizer/Square/ReadVariableOp!^conv3d_7/StatefulPartitionedCall2^conv3d_7/kernel/Regularizer/Square/ReadVariableOp^dense/StatefulPartitionedCall/^dense/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:���������22: : : : : : : : : : : : : : : : : : 2@
conv3d/StatefulPartitionedCallconv3d/StatefulPartitionedCall2b
/conv3d/kernel/Regularizer/Square/ReadVariableOp/conv3d/kernel/Regularizer/Square/ReadVariableOp2D
 conv3d_1/StatefulPartitionedCall conv3d_1/StatefulPartitionedCall2f
1conv3d_1/kernel/Regularizer/Square/ReadVariableOp1conv3d_1/kernel/Regularizer/Square/ReadVariableOp2D
 conv3d_2/StatefulPartitionedCall conv3d_2/StatefulPartitionedCall2f
1conv3d_2/kernel/Regularizer/Square/ReadVariableOp1conv3d_2/kernel/Regularizer/Square/ReadVariableOp2D
 conv3d_3/StatefulPartitionedCall conv3d_3/StatefulPartitionedCall2f
1conv3d_3/kernel/Regularizer/Square/ReadVariableOp1conv3d_3/kernel/Regularizer/Square/ReadVariableOp2D
 conv3d_4/StatefulPartitionedCall conv3d_4/StatefulPartitionedCall2f
1conv3d_4/kernel/Regularizer/Square/ReadVariableOp1conv3d_4/kernel/Regularizer/Square/ReadVariableOp2D
 conv3d_5/StatefulPartitionedCall conv3d_5/StatefulPartitionedCall2f
1conv3d_5/kernel/Regularizer/Square/ReadVariableOp1conv3d_5/kernel/Regularizer/Square/ReadVariableOp2D
 conv3d_6/StatefulPartitionedCall conv3d_6/StatefulPartitionedCall2f
1conv3d_6/kernel/Regularizer/Square/ReadVariableOp1conv3d_6/kernel/Regularizer/Square/ReadVariableOp2D
 conv3d_7/StatefulPartitionedCall conv3d_7/StatefulPartitionedCall2f
1conv3d_7/kernel/Regularizer/Square/ReadVariableOp1conv3d_7/kernel/Regularizer/Square/ReadVariableOp2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2`
.dense/kernel/Regularizer/Square/ReadVariableOp.dense/kernel/Regularizer/Square/ReadVariableOp:\ X
3
_output_shapes!
:���������22
!
_user_specified_name	input_1
�
�
*__inference_add_metric_layer_call_fn_16833

inputs
unknown: 
	unknown_0: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
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
GPU(2*0J 8� *N
fIRG
E__inference_add_metric_layer_call_and_return_conditional_losses_14235^
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes
: `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : 22
StatefulPartitionedCallStatefulPartitionedCall:> :

_output_shapes
: 
 
_user_specified_nameinputs
��
�
B__inference_model_1_layer_call_and_return_conditional_losses_15162
input_2
input_3
input_4
input_5)
model_14897:@
model_14899:@)
model_14901:@@
model_14903:@*
model_14905:@�
model_14907:	�+
model_14909:��
model_14911:	�+
model_14913:��
model_14915:	�+
model_14917:��
model_14919:	�+
model_14921:��
model_14923:	�+
model_14925:��
model_14927:	�
model_14929:	�$
model_14931:
add_metric_15068: 
add_metric_15070: 
unknown
	unknown_0
	unknown_1
add_metric_1_15100: 
add_metric_1_15102: 
identity

identity_1

identity_2��"add_metric/StatefulPartitionedCall�$add_metric_1/StatefulPartitionedCall�/conv3d/kernel/Regularizer/Square/ReadVariableOp�1conv3d_1/kernel/Regularizer/Square/ReadVariableOp�1conv3d_2/kernel/Regularizer/Square/ReadVariableOp�1conv3d_3/kernel/Regularizer/Square/ReadVariableOp�1conv3d_4/kernel/Regularizer/Square/ReadVariableOp�1conv3d_5/kernel/Regularizer/Square/ReadVariableOp�1conv3d_6/kernel/Regularizer/Square/ReadVariableOp�1conv3d_7/kernel/Regularizer/Square/ReadVariableOp�.dense/kernel/Regularizer/Square/ReadVariableOp�model/StatefulPartitionedCall�model/StatefulPartitionedCall_1�
model/StatefulPartitionedCallStatefulPartitionedCallinput_3model_14897model_14899model_14901model_14903model_14905model_14907model_14909model_14911model_14913model_14915model_14917model_14919model_14921model_14923model_14925model_14927model_14929model_14931*
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
@__inference_model_layer_call_and_return_conditional_losses_13452�
model/StatefulPartitionedCall_1StatefulPartitionedCallinput_2model_14897model_14899model_14901model_14903model_14905model_14907model_14909model_14911model_14913model_14915model_14917model_14919model_14921model_14923model_14925model_14927model_14929model_14931*
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
@__inference_model_layer_call_and_return_conditional_losses_13452U
tf.compat.v1.squeeze_1/SqueezeSqueezeinput_5*
T0*
_output_shapes
:t
tf.compat.v1.squeeze_3/SqueezeSqueeze&model/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:S
tf.compat.v1.squeeze/SqueezeSqueezeinput_4*
T0*
_output_shapes
:v
tf.compat.v1.squeeze_2/SqueezeSqueeze(model/StatefulPartitionedCall_1:output:0*
T0*
_output_shapes
:v
tf.compat.v1.squeeze_4/SqueezeSqueeze(model/StatefulPartitionedCall_1:output:0*
T0*
_output_shapes
:t
tf.compat.v1.squeeze_5/SqueezeSqueeze&model/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:U
tf.compat.v1.squeeze_7/SqueezeSqueezeinput_5*
T0*
_output_shapes
:t
tf.compat.v1.squeeze_9/SqueezeSqueeze&model/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:U
tf.compat.v1.squeeze_6/SqueezeSqueezeinput_4*
T0*
_output_shapes
:v
tf.compat.v1.squeeze_8/SqueezeSqueeze(model/StatefulPartitionedCall_1:output:0*
T0*
_output_shapes
:u
tf.compat.v1.squeeze_13/SqueezeSqueeze&model/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:V
tf.compat.v1.squeeze_12/SqueezeSqueezeinput_5*
T0*
_output_shapes
:w
tf.compat.v1.squeeze_11/SqueezeSqueeze(model/StatefulPartitionedCall_1:output:0*
T0*
_output_shapes
:V
tf.compat.v1.squeeze_10/SqueezeSqueezeinput_4*
T0*
_output_shapes
:�
tf.math.subtract_1/SubSub'tf.compat.v1.squeeze_1/Squeeze:output:0'tf.compat.v1.squeeze_3/Squeeze:output:0*
T0*
_output_shapes
:�
tf.math.subtract/SubSub%tf.compat.v1.squeeze/Squeeze:output:0'tf.compat.v1.squeeze_2/Squeeze:output:0*
T0*
_output_shapes
:�
tf.math.subtract_2/SubSub'tf.compat.v1.squeeze_4/Squeeze:output:0'tf.compat.v1.squeeze_5/Squeeze:output:0*
T0*
_output_shapes
:�
tf.math.subtract_5/SubSub'tf.compat.v1.squeeze_7/Squeeze:output:0'tf.compat.v1.squeeze_9/Squeeze:output:0*
T0*
_output_shapes
:�
tf.math.subtract_4/SubSub'tf.compat.v1.squeeze_6/Squeeze:output:0'tf.compat.v1.squeeze_8/Squeeze:output:0*
T0*
_output_shapes
:m
tf.math.reduce_mean_8/RankRank(tf.compat.v1.squeeze_13/Squeeze:output:0*
T0*
_output_shapes
: c
!tf.math.reduce_mean_8/range/startConst*
_output_shapes
: *
dtype0*
value	B : c
!tf.math.reduce_mean_8/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
tf.math.reduce_mean_8/rangeRange*tf.math.reduce_mean_8/range/start:output:0#tf.math.reduce_mean_8/Rank:output:0*tf.math.reduce_mean_8/range/delta:output:0*#
_output_shapes
:����������
tf.math.reduce_mean_8/MeanMean(tf.compat.v1.squeeze_13/Squeeze:output:0$tf.math.reduce_mean_8/range:output:0*
T0*
_output_shapes
: m
tf.math.reduce_mean_7/RankRank(tf.compat.v1.squeeze_12/Squeeze:output:0*
T0*
_output_shapes
: c
!tf.math.reduce_mean_7/range/startConst*
_output_shapes
: *
dtype0*
value	B : c
!tf.math.reduce_mean_7/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
tf.math.reduce_mean_7/rangeRange*tf.math.reduce_mean_7/range/start:output:0#tf.math.reduce_mean_7/Rank:output:0*tf.math.reduce_mean_7/range/delta:output:0*#
_output_shapes
:����������
tf.math.reduce_mean_7/MeanMean(tf.compat.v1.squeeze_12/Squeeze:output:0$tf.math.reduce_mean_7/range:output:0*
T0*
_output_shapes
: m
tf.math.reduce_mean_6/RankRank(tf.compat.v1.squeeze_11/Squeeze:output:0*
T0*
_output_shapes
: c
!tf.math.reduce_mean_6/range/startConst*
_output_shapes
: *
dtype0*
value	B : c
!tf.math.reduce_mean_6/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
tf.math.reduce_mean_6/rangeRange*tf.math.reduce_mean_6/range/start:output:0#tf.math.reduce_mean_6/Rank:output:0*tf.math.reduce_mean_6/range/delta:output:0*#
_output_shapes
:����������
tf.math.reduce_mean_6/MeanMean(tf.compat.v1.squeeze_11/Squeeze:output:0$tf.math.reduce_mean_6/range:output:0*
T0*
_output_shapes
: m
tf.math.reduce_mean_5/RankRank(tf.compat.v1.squeeze_10/Squeeze:output:0*
T0*
_output_shapes
: c
!tf.math.reduce_mean_5/range/startConst*
_output_shapes
: *
dtype0*
value	B : c
!tf.math.reduce_mean_5/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
tf.math.reduce_mean_5/rangeRange*tf.math.reduce_mean_5/range/start:output:0#tf.math.reduce_mean_5/Rank:output:0*tf.math.reduce_mean_5/range/delta:output:0*#
_output_shapes
:����������
tf.math.reduce_mean_5/MeanMean(tf.compat.v1.squeeze_10/Squeeze:output:0$tf.math.reduce_mean_5/range:output:0*
T0*
_output_shapes
: `
tf.math.square_1/SquareSquaretf.math.subtract_1/Sub:z:0*
T0*
_output_shapes
:\
tf.math.square/SquareSquaretf.math.subtract/Sub:z:0*
T0*
_output_shapes
:`
tf.math.square_2/SquareSquaretf.math.subtract_2/Sub:z:0*
T0*
_output_shapes
:`
tf.math.square_4/SquareSquaretf.math.subtract_5/Sub:z:0*
T0*
_output_shapes
:`
tf.math.square_3/SquareSquaretf.math.subtract_4/Sub:z:0*
T0*
_output_shapes
:�
tf.math.subtract_10/SubSub(tf.compat.v1.squeeze_13/Squeeze:output:0#tf.math.reduce_mean_8/Mean:output:0*
T0*
_output_shapes
:�
tf.math.subtract_9/SubSub(tf.compat.v1.squeeze_12/Squeeze:output:0#tf.math.reduce_mean_7/Mean:output:0*
T0*
_output_shapes
:�
tf.math.subtract_7/SubSub(tf.compat.v1.squeeze_11/Squeeze:output:0#tf.math.reduce_mean_6/Mean:output:0*
T0*
_output_shapes
:�
tf.math.subtract_6/SubSub(tf.compat.v1.squeeze_10/Squeeze:output:0#tf.math.reduce_mean_5/Mean:output:0*
T0*
_output_shapes
:\
tf.math.reduce_mean/RankRanktf.math.square/Square:y:0*
T0*
_output_shapes
: a
tf.math.reduce_mean/range/startConst*
_output_shapes
: *
dtype0*
value	B : a
tf.math.reduce_mean/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
tf.math.reduce_mean/rangeRange(tf.math.reduce_mean/range/start:output:0!tf.math.reduce_mean/Rank:output:0(tf.math.reduce_mean/range/delta:output:0*#
_output_shapes
:����������
tf.math.reduce_mean/MeanMeantf.math.square/Square:y:0"tf.math.reduce_mean/range:output:0*
T0*
_output_shapes
: `
tf.math.reduce_mean_1/RankRanktf.math.square_1/Square:y:0*
T0*
_output_shapes
: c
!tf.math.reduce_mean_1/range/startConst*
_output_shapes
: *
dtype0*
value	B : c
!tf.math.reduce_mean_1/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
tf.math.reduce_mean_1/rangeRange*tf.math.reduce_mean_1/range/start:output:0#tf.math.reduce_mean_1/Rank:output:0*tf.math.reduce_mean_1/range/delta:output:0*#
_output_shapes
:����������
tf.math.reduce_mean_1/MeanMeantf.math.square_1/Square:y:0$tf.math.reduce_mean_1/range:output:0*
T0*
_output_shapes
: ^
tf.math.maximum/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *��:�
tf.math.maximum/MaximumMaximumtf.math.square_2/Square:y:0"tf.math.maximum/Maximum/y:output:0*
T0*
_output_shapes
:`
tf.math.reduce_mean_3/RankRanktf.math.square_3/Square:y:0*
T0*
_output_shapes
: c
!tf.math.reduce_mean_3/range/startConst*
_output_shapes
: *
dtype0*
value	B : c
!tf.math.reduce_mean_3/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
tf.math.reduce_mean_3/rangeRange*tf.math.reduce_mean_3/range/start:output:0#tf.math.reduce_mean_3/Rank:output:0*tf.math.reduce_mean_3/range/delta:output:0*#
_output_shapes
:����������
tf.math.reduce_mean_3/MeanMeantf.math.square_3/Square:y:0$tf.math.reduce_mean_3/range:output:0*
T0*
_output_shapes
: `
tf.math.reduce_mean_4/RankRanktf.math.square_4/Square:y:0*
T0*
_output_shapes
: c
!tf.math.reduce_mean_4/range/startConst*
_output_shapes
: *
dtype0*
value	B : c
!tf.math.reduce_mean_4/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
tf.math.reduce_mean_4/rangeRange*tf.math.reduce_mean_4/range/start:output:0#tf.math.reduce_mean_4/Rank:output:0*tf.math.reduce_mean_4/range/delta:output:0*#
_output_shapes
:����������
tf.math.reduce_mean_4/MeanMeantf.math.square_4/Square:y:0$tf.math.reduce_mean_4/range:output:0*
T0*
_output_shapes
: y
tf.math.multiply_3/MulMultf.math.subtract_9/Sub:z:0tf.math.subtract_10/Sub:z:0*
T0*
_output_shapes
:x
tf.math.multiply_1/MulMultf.math.subtract_6/Sub:z:0tf.math.subtract_7/Sub:z:0*
T0*
_output_shapes
:a
tf.math.square_8/SquareSquaretf.math.subtract_10/Sub:z:0*
T0*
_output_shapes
:`
tf.math.square_7/SquareSquaretf.math.subtract_9/Sub:z:0*
T0*
_output_shapes
:`
tf.math.square_6/SquareSquaretf.math.subtract_7/Sub:z:0*
T0*
_output_shapes
:`
tf.math.square_5/SquareSquaretf.math.subtract_6/Sub:z:0*
T0*
_output_shapes
:�
tf.__operators__.add/AddV2AddV2!tf.math.reduce_mean/Mean:output:0#tf.math.reduce_mean_1/Mean:output:0*
T0*
_output_shapes
: ]
tf.math.subtract_3/Sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *��:�
tf.math.subtract_3/SubSubtf.math.maximum/Maximum:z:0!tf.math.subtract_3/Sub/y:output:0*
T0*
_output_shapes
:�
tf.__operators__.add_2/AddV2AddV2#tf.math.reduce_mean_3/Mean:output:0#tf.math.reduce_mean_4/Mean:output:0*
T0*
_output_shapes
: ^
tf.math.reduce_sum_3/RankRanktf.math.multiply_3/Mul:z:0*
T0*
_output_shapes
: b
 tf.math.reduce_sum_3/range/startConst*
_output_shapes
: *
dtype0*
value	B : b
 tf.math.reduce_sum_3/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
tf.math.reduce_sum_3/rangeRange)tf.math.reduce_sum_3/range/start:output:0"tf.math.reduce_sum_3/Rank:output:0)tf.math.reduce_sum_3/range/delta:output:0*#
_output_shapes
:����������
tf.math.reduce_sum_3/SumSumtf.math.multiply_3/Mul:z:0#tf.math.reduce_sum_3/range:output:0*
T0*
_output_shapes
: \
tf.math.reduce_sum/RankRanktf.math.multiply_1/Mul:z:0*
T0*
_output_shapes
: `
tf.math.reduce_sum/range/startConst*
_output_shapes
: *
dtype0*
value	B : `
tf.math.reduce_sum/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
tf.math.reduce_sum/rangeRange'tf.math.reduce_sum/range/start:output:0 tf.math.reduce_sum/Rank:output:0'tf.math.reduce_sum/range/delta:output:0*#
_output_shapes
:���������}
tf.math.reduce_sum/SumSumtf.math.multiply_1/Mul:z:0!tf.math.reduce_sum/range:output:0*
T0*
_output_shapes
: _
tf.math.reduce_sum_4/RankRanktf.math.square_7/Square:y:0*
T0*
_output_shapes
: b
 tf.math.reduce_sum_4/range/startConst*
_output_shapes
: *
dtype0*
value	B : b
 tf.math.reduce_sum_4/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
tf.math.reduce_sum_4/rangeRange)tf.math.reduce_sum_4/range/start:output:0"tf.math.reduce_sum_4/Rank:output:0)tf.math.reduce_sum_4/range/delta:output:0*#
_output_shapes
:����������
tf.math.reduce_sum_4/SumSumtf.math.square_7/Square:y:0#tf.math.reduce_sum_4/range:output:0*
T0*
_output_shapes
: _
tf.math.reduce_sum_5/RankRanktf.math.square_8/Square:y:0*
T0*
_output_shapes
: b
 tf.math.reduce_sum_5/range/startConst*
_output_shapes
: *
dtype0*
value	B : b
 tf.math.reduce_sum_5/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
tf.math.reduce_sum_5/rangeRange)tf.math.reduce_sum_5/range/start:output:0"tf.math.reduce_sum_5/Rank:output:0)tf.math.reduce_sum_5/range/delta:output:0*#
_output_shapes
:����������
tf.math.reduce_sum_5/SumSumtf.math.square_8/Square:y:0#tf.math.reduce_sum_5/range:output:0*
T0*
_output_shapes
: _
tf.math.reduce_sum_1/RankRanktf.math.square_5/Square:y:0*
T0*
_output_shapes
: b
 tf.math.reduce_sum_1/range/startConst*
_output_shapes
: *
dtype0*
value	B : b
 tf.math.reduce_sum_1/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
tf.math.reduce_sum_1/rangeRange)tf.math.reduce_sum_1/range/start:output:0"tf.math.reduce_sum_1/Rank:output:0)tf.math.reduce_sum_1/range/delta:output:0*#
_output_shapes
:����������
tf.math.reduce_sum_1/SumSumtf.math.square_5/Square:y:0#tf.math.reduce_sum_1/range:output:0*
T0*
_output_shapes
: _
tf.math.reduce_sum_2/RankRanktf.math.square_6/Square:y:0*
T0*
_output_shapes
: b
 tf.math.reduce_sum_2/range/startConst*
_output_shapes
: *
dtype0*
value	B : b
 tf.math.reduce_sum_2/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
tf.math.reduce_sum_2/rangeRange)tf.math.reduce_sum_2/range/start:output:0"tf.math.reduce_sum_2/Rank:output:0)tf.math.reduce_sum_2/range/delta:output:0*#
_output_shapes
:����������
tf.math.reduce_sum_2/SumSumtf.math.square_6/Square:y:0#tf.math.reduce_sum_2/range:output:0*
T0*
_output_shapes
: _
tf.math.reduce_mean_2/RankRanktf.math.subtract_3/Sub:z:0*
T0*
_output_shapes
: c
!tf.math.reduce_mean_2/range/startConst*
_output_shapes
: *
dtype0*
value	B : c
!tf.math.reduce_mean_2/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
tf.math.reduce_mean_2/rangeRange*tf.math.reduce_mean_2/range/start:output:0#tf.math.reduce_mean_2/Rank:output:0*tf.math.reduce_mean_2/range/delta:output:0*#
_output_shapes
:����������
tf.math.reduce_mean_2/MeanMeantf.math.subtract_3/Sub:z:0$tf.math.reduce_mean_2/range:output:0*
T0*
_output_shapes
: �
"add_metric/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_2/AddV2:z:0add_metric_15068add_metric_15070*
Tin
2*
Tout
2*
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
GPU(2*0J 8� *N
fIRG
E__inference_add_metric_layer_call_and_return_conditional_losses_14235�
tf.math.multiply_4/MulMul!tf.math.reduce_sum_4/Sum:output:0!tf.math.reduce_sum_5/Sum:output:0*
T0*
_output_shapes
: �
tf.math.multiply_2/MulMul!tf.math.reduce_sum_1/Sum:output:0!tf.math.reduce_sum_2/Sum:output:0*
T0*
_output_shapes
: j
tf.math.multiply/MulMulunknown#tf.math.reduce_mean_2/Mean:output:0*
T0*
_output_shapes
: X
tf.math.sqrt_1/SqrtSqrttf.math.multiply_4/Mul:z:0*
T0*
_output_shapes
: V
tf.math.sqrt/SqrtSqrttf.math.multiply_2/Mul:z:0*
T0*
_output_shapes
: �
tf.__operators__.add_1/AddV2AddV2tf.__operators__.add/AddV2:z:0tf.math.multiply/Mul:z:0*
T0*
_output_shapes
: ]
tf.__operators__.add_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *��'7�
tf.__operators__.add_4/AddV2AddV2tf.math.sqrt_1/Sqrt:y:0!tf.__operators__.add_4/y:output:0*
T0*
_output_shapes
: ]
tf.__operators__.add_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *��'7�
tf.__operators__.add_3/AddV2AddV2tf.math.sqrt/Sqrt:y:0!tf.__operators__.add_3/y:output:0*
T0*
_output_shapes
: �
add_loss/PartitionedCallPartitionedCall tf.__operators__.add_1/AddV2:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: : * 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8� *L
fGRE
C__inference_add_loss_layer_call_and_return_conditional_losses_14258�
tf.math.truediv_1/truedivRealDiv!tf.math.reduce_sum_3/Sum:output:0 tf.__operators__.add_4/AddV2:z:0*
T0*
_output_shapes
: �
tf.math.truediv/truedivRealDivtf.math.reduce_sum/Sum:output:0 tf.__operators__.add_3/AddV2:z:0*
T0*
_output_shapes
: i
tf.math.subtract_11/SubSub	unknown_0tf.math.truediv_1/truediv:z:0*
T0*
_output_shapes
: f
tf.math.subtract_8/SubSub	unknown_1tf.math.truediv/truediv:z:0*
T0*
_output_shapes
: V
tf.math.pow/Pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?o
tf.math.pow/PowPowtf.math.subtract_8/Sub:z:0tf.math.pow/Pow/y:output:0*
T0*
_output_shapes
: X
tf.math.pow_1/Pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?t
tf.math.pow_1/PowPowtf.math.subtract_11/Sub:z:0tf.math.pow_1/Pow/y:output:0*
T0*
_output_shapes
: r
tf.__operators__.add_5/AddV2AddV2tf.math.pow/Pow:z:0tf.math.pow_1/Pow:z:0*
T0*
_output_shapes
: �
$add_metric_1/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_5/AddV2:z:0add_metric_1_15100add_metric_1_15102*
Tin
2*
Tout
2*
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
GPU(2*0J 8� *P
fKRI
G__inference_add_metric_1_layer_call_and_return_conditional_losses_14291�
/conv3d/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmodel_14897**
_output_shapes
:@*
dtype0�
 conv3d/kernel/Regularizer/SquareSquare7conv3d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0**
_output_shapes
:@|
conv3d/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
conv3d/kernel/Regularizer/SumSum$conv3d/kernel/Regularizer/Square:y:0(conv3d/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: d
conv3d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    �
conv3d/kernel/Regularizer/mulMul(conv3d/kernel/Regularizer/mul/x:output:0&conv3d/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1conv3d_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmodel_14901**
_output_shapes
:@@*
dtype0�
"conv3d_1/kernel/Regularizer/SquareSquare9conv3d_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0**
_output_shapes
:@@~
!conv3d_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
conv3d_1/kernel/Regularizer/SumSum&conv3d_1/kernel/Regularizer/Square:y:0*conv3d_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!conv3d_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    �
conv3d_1/kernel/Regularizer/mulMul*conv3d_1/kernel/Regularizer/mul/x:output:0(conv3d_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1conv3d_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmodel_14905*+
_output_shapes
:@�*
dtype0�
"conv3d_2/kernel/Regularizer/SquareSquare9conv3d_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*+
_output_shapes
:@�~
!conv3d_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
conv3d_2/kernel/Regularizer/SumSum&conv3d_2/kernel/Regularizer/Square:y:0*conv3d_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!conv3d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    �
conv3d_2/kernel/Regularizer/mulMul*conv3d_2/kernel/Regularizer/mul/x:output:0(conv3d_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1conv3d_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmodel_14909*,
_output_shapes
:��*
dtype0�
"conv3d_3/kernel/Regularizer/SquareSquare9conv3d_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*,
_output_shapes
:��~
!conv3d_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
conv3d_3/kernel/Regularizer/SumSum&conv3d_3/kernel/Regularizer/Square:y:0*conv3d_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!conv3d_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    �
conv3d_3/kernel/Regularizer/mulMul*conv3d_3/kernel/Regularizer/mul/x:output:0(conv3d_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1conv3d_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmodel_14913*,
_output_shapes
:��*
dtype0�
"conv3d_4/kernel/Regularizer/SquareSquare9conv3d_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*,
_output_shapes
:��~
!conv3d_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
conv3d_4/kernel/Regularizer/SumSum&conv3d_4/kernel/Regularizer/Square:y:0*conv3d_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!conv3d_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    �
conv3d_4/kernel/Regularizer/mulMul*conv3d_4/kernel/Regularizer/mul/x:output:0(conv3d_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1conv3d_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmodel_14917*,
_output_shapes
:��*
dtype0�
"conv3d_5/kernel/Regularizer/SquareSquare9conv3d_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*,
_output_shapes
:��~
!conv3d_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
conv3d_5/kernel/Regularizer/SumSum&conv3d_5/kernel/Regularizer/Square:y:0*conv3d_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!conv3d_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    �
conv3d_5/kernel/Regularizer/mulMul*conv3d_5/kernel/Regularizer/mul/x:output:0(conv3d_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1conv3d_6/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmodel_14921*,
_output_shapes
:��*
dtype0�
"conv3d_6/kernel/Regularizer/SquareSquare9conv3d_6/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*,
_output_shapes
:��~
!conv3d_6/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
conv3d_6/kernel/Regularizer/SumSum&conv3d_6/kernel/Regularizer/Square:y:0*conv3d_6/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!conv3d_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    �
conv3d_6/kernel/Regularizer/mulMul*conv3d_6/kernel/Regularizer/mul/x:output:0(conv3d_6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1conv3d_7/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmodel_14925*,
_output_shapes
:��*
dtype0�
"conv3d_7/kernel/Regularizer/SquareSquare9conv3d_7/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*,
_output_shapes
:��~
!conv3d_7/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
conv3d_7/kernel/Regularizer/SumSum&conv3d_7/kernel/Regularizer/Square:y:0*conv3d_7/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!conv3d_7/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    �
conv3d_7/kernel/Regularizer/mulMul*conv3d_7/kernel/Regularizer/mul/x:output:0(conv3d_7/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: {
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmodel_14929*
_output_shapes
:	�$*
dtype0�
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	�$o
dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
dense/kernel/Regularizer/SumSum#dense/kernel/Regularizer/Square:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: c
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    �
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: w
IdentityIdentity(model/StatefulPartitionedCall_1:output:0^NoOp*
T0*'
_output_shapes
:���������w

Identity_1Identity&model/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������a

Identity_2Identity!add_loss/PartitionedCall:output:1^NoOp*
T0*
_output_shapes
: �
NoOpNoOp#^add_metric/StatefulPartitionedCall%^add_metric_1/StatefulPartitionedCall0^conv3d/kernel/Regularizer/Square/ReadVariableOp2^conv3d_1/kernel/Regularizer/Square/ReadVariableOp2^conv3d_2/kernel/Regularizer/Square/ReadVariableOp2^conv3d_3/kernel/Regularizer/Square/ReadVariableOp2^conv3d_4/kernel/Regularizer/Square/ReadVariableOp2^conv3d_5/kernel/Regularizer/Square/ReadVariableOp2^conv3d_6/kernel/Regularizer/Square/ReadVariableOp2^conv3d_7/kernel/Regularizer/Square/ReadVariableOp/^dense/kernel/Regularizer/Square/ReadVariableOp^model/StatefulPartitionedCall ^model/StatefulPartitionedCall_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������22:���������22:���������:���������: : : : : : : : : : : : : : : : : : : : : : : : : 2H
"add_metric/StatefulPartitionedCall"add_metric/StatefulPartitionedCall2L
$add_metric_1/StatefulPartitionedCall$add_metric_1/StatefulPartitionedCall2b
/conv3d/kernel/Regularizer/Square/ReadVariableOp/conv3d/kernel/Regularizer/Square/ReadVariableOp2f
1conv3d_1/kernel/Regularizer/Square/ReadVariableOp1conv3d_1/kernel/Regularizer/Square/ReadVariableOp2f
1conv3d_2/kernel/Regularizer/Square/ReadVariableOp1conv3d_2/kernel/Regularizer/Square/ReadVariableOp2f
1conv3d_3/kernel/Regularizer/Square/ReadVariableOp1conv3d_3/kernel/Regularizer/Square/ReadVariableOp2f
1conv3d_4/kernel/Regularizer/Square/ReadVariableOp1conv3d_4/kernel/Regularizer/Square/ReadVariableOp2f
1conv3d_5/kernel/Regularizer/Square/ReadVariableOp1conv3d_5/kernel/Regularizer/Square/ReadVariableOp2f
1conv3d_6/kernel/Regularizer/Square/ReadVariableOp1conv3d_6/kernel/Regularizer/Square/ReadVariableOp2f
1conv3d_7/kernel/Regularizer/Square/ReadVariableOp1conv3d_7/kernel/Regularizer/Square/ReadVariableOp2`
.dense/kernel/Regularizer/Square/ReadVariableOp.dense/kernel/Regularizer/Square/ReadVariableOp2>
model/StatefulPartitionedCallmodel/StatefulPartitionedCall2B
model/StatefulPartitionedCall_1model/StatefulPartitionedCall_1:\ X
3
_output_shapes!
:���������22
!
_user_specified_name	input_2:\X
3
_output_shapes!
:���������22
!
_user_specified_name	input_3:PL
'
_output_shapes
:���������
!
_user_specified_name	input_4:PL
'
_output_shapes
:���������
!
_user_specified_name	input_5:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
�
@__inference_model_layer_call_and_return_conditional_losses_14034
input_1*
conv3d_13929:@
conv3d_13931:@,
conv3d_1_13934:@@
conv3d_1_13936:@-
conv3d_2_13940:@�
conv3d_2_13942:	�.
conv3d_3_13945:��
conv3d_3_13947:	�.
conv3d_4_13951:��
conv3d_4_13953:	�.
conv3d_5_13956:��
conv3d_5_13958:	�.
conv3d_6_13962:��
conv3d_6_13964:	�.
conv3d_7_13967:��
conv3d_7_13969:	�
dense_13974:	�$
dense_13976:
identity��conv3d/StatefulPartitionedCall�/conv3d/kernel/Regularizer/Square/ReadVariableOp� conv3d_1/StatefulPartitionedCall�1conv3d_1/kernel/Regularizer/Square/ReadVariableOp� conv3d_2/StatefulPartitionedCall�1conv3d_2/kernel/Regularizer/Square/ReadVariableOp� conv3d_3/StatefulPartitionedCall�1conv3d_3/kernel/Regularizer/Square/ReadVariableOp� conv3d_4/StatefulPartitionedCall�1conv3d_4/kernel/Regularizer/Square/ReadVariableOp� conv3d_5/StatefulPartitionedCall�1conv3d_5/kernel/Regularizer/Square/ReadVariableOp� conv3d_6/StatefulPartitionedCall�1conv3d_6/kernel/Regularizer/Square/ReadVariableOp� conv3d_7/StatefulPartitionedCall�1conv3d_7/kernel/Regularizer/Square/ReadVariableOp�dense/StatefulPartitionedCall�.dense/kernel/Regularizer/Square/ReadVariableOp�
conv3d/StatefulPartitionedCallStatefulPartitionedCallinput_1conv3d_13929conv3d_13931*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������22@*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *J
fERC
A__inference_conv3d_layer_call_and_return_conditional_losses_13196�
 conv3d_1/StatefulPartitionedCallStatefulPartitionedCall'conv3d/StatefulPartitionedCall:output:0conv3d_1_13934conv3d_1_13936*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������22@*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *L
fGRE
C__inference_conv3d_1_layer_call_and_return_conditional_losses_13219�
max_pooling3d/PartitionedCallPartitionedCall)conv3d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������
@* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8� *Q
fLRJ
H__inference_max_pooling3d_layer_call_and_return_conditional_losses_13133�
 conv3d_2/StatefulPartitionedCallStatefulPartitionedCall&max_pooling3d/PartitionedCall:output:0conv3d_2_13940conv3d_2_13942*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :���������
�*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *L
fGRE
C__inference_conv3d_2_layer_call_and_return_conditional_losses_13243�
 conv3d_3/StatefulPartitionedCallStatefulPartitionedCall)conv3d_2/StatefulPartitionedCall:output:0conv3d_3_13945conv3d_3_13947*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :���������
�*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *L
fGRE
C__inference_conv3d_3_layer_call_and_return_conditional_losses_13266�
max_pooling3d_1/PartitionedCallPartitionedCall)conv3d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :����������* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8� *S
fNRL
J__inference_max_pooling3d_1_layer_call_and_return_conditional_losses_13145�
 conv3d_4/StatefulPartitionedCallStatefulPartitionedCall(max_pooling3d_1/PartitionedCall:output:0conv3d_4_13951conv3d_4_13953*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :����������*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *L
fGRE
C__inference_conv3d_4_layer_call_and_return_conditional_losses_13290�
 conv3d_5/StatefulPartitionedCallStatefulPartitionedCall)conv3d_4/StatefulPartitionedCall:output:0conv3d_5_13956conv3d_5_13958*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :����������*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *L
fGRE
C__inference_conv3d_5_layer_call_and_return_conditional_losses_13313�
max_pooling3d_2/PartitionedCallPartitionedCall)conv3d_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :����������* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8� *S
fNRL
J__inference_max_pooling3d_2_layer_call_and_return_conditional_losses_13157�
 conv3d_6/StatefulPartitionedCallStatefulPartitionedCall(max_pooling3d_2/PartitionedCall:output:0conv3d_6_13962conv3d_6_13964*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :����������*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *L
fGRE
C__inference_conv3d_6_layer_call_and_return_conditional_losses_13337�
 conv3d_7/StatefulPartitionedCallStatefulPartitionedCall)conv3d_6/StatefulPartitionedCall:output:0conv3d_7_13967conv3d_7_13969*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :����������*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *L
fGRE
C__inference_conv3d_7_layer_call_and_return_conditional_losses_13360�
max_pooling3d_3/PartitionedCallPartitionedCall)conv3d_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :����������* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8� *S
fNRL
J__inference_max_pooling3d_3_layer_call_and_return_conditional_losses_13169�
flatten/PartitionedCallPartitionedCall(max_pooling3d_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������$* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8� *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_13373�
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_13974dense_13976*
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
@__inference_dense_layer_call_and_return_conditional_losses_13391�
/conv3d/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv3d_13929**
_output_shapes
:@*
dtype0�
 conv3d/kernel/Regularizer/SquareSquare7conv3d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0**
_output_shapes
:@|
conv3d/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
conv3d/kernel/Regularizer/SumSum$conv3d/kernel/Regularizer/Square:y:0(conv3d/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: d
conv3d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    �
conv3d/kernel/Regularizer/mulMul(conv3d/kernel/Regularizer/mul/x:output:0&conv3d/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1conv3d_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv3d_1_13934**
_output_shapes
:@@*
dtype0�
"conv3d_1/kernel/Regularizer/SquareSquare9conv3d_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0**
_output_shapes
:@@~
!conv3d_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
conv3d_1/kernel/Regularizer/SumSum&conv3d_1/kernel/Regularizer/Square:y:0*conv3d_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!conv3d_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    �
conv3d_1/kernel/Regularizer/mulMul*conv3d_1/kernel/Regularizer/mul/x:output:0(conv3d_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1conv3d_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv3d_2_13940*+
_output_shapes
:@�*
dtype0�
"conv3d_2/kernel/Regularizer/SquareSquare9conv3d_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*+
_output_shapes
:@�~
!conv3d_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
conv3d_2/kernel/Regularizer/SumSum&conv3d_2/kernel/Regularizer/Square:y:0*conv3d_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!conv3d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    �
conv3d_2/kernel/Regularizer/mulMul*conv3d_2/kernel/Regularizer/mul/x:output:0(conv3d_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1conv3d_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv3d_3_13945*,
_output_shapes
:��*
dtype0�
"conv3d_3/kernel/Regularizer/SquareSquare9conv3d_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*,
_output_shapes
:��~
!conv3d_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
conv3d_3/kernel/Regularizer/SumSum&conv3d_3/kernel/Regularizer/Square:y:0*conv3d_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!conv3d_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    �
conv3d_3/kernel/Regularizer/mulMul*conv3d_3/kernel/Regularizer/mul/x:output:0(conv3d_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1conv3d_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv3d_4_13951*,
_output_shapes
:��*
dtype0�
"conv3d_4/kernel/Regularizer/SquareSquare9conv3d_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*,
_output_shapes
:��~
!conv3d_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
conv3d_4/kernel/Regularizer/SumSum&conv3d_4/kernel/Regularizer/Square:y:0*conv3d_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!conv3d_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    �
conv3d_4/kernel/Regularizer/mulMul*conv3d_4/kernel/Regularizer/mul/x:output:0(conv3d_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1conv3d_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv3d_5_13956*,
_output_shapes
:��*
dtype0�
"conv3d_5/kernel/Regularizer/SquareSquare9conv3d_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*,
_output_shapes
:��~
!conv3d_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
conv3d_5/kernel/Regularizer/SumSum&conv3d_5/kernel/Regularizer/Square:y:0*conv3d_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!conv3d_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    �
conv3d_5/kernel/Regularizer/mulMul*conv3d_5/kernel/Regularizer/mul/x:output:0(conv3d_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1conv3d_6/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv3d_6_13962*,
_output_shapes
:��*
dtype0�
"conv3d_6/kernel/Regularizer/SquareSquare9conv3d_6/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*,
_output_shapes
:��~
!conv3d_6/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
conv3d_6/kernel/Regularizer/SumSum&conv3d_6/kernel/Regularizer/Square:y:0*conv3d_6/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!conv3d_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    �
conv3d_6/kernel/Regularizer/mulMul*conv3d_6/kernel/Regularizer/mul/x:output:0(conv3d_6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1conv3d_7/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv3d_7_13967*,
_output_shapes
:��*
dtype0�
"conv3d_7/kernel/Regularizer/SquareSquare9conv3d_7/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*,
_output_shapes
:��~
!conv3d_7/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
conv3d_7/kernel/Regularizer/SumSum&conv3d_7/kernel/Regularizer/Square:y:0*conv3d_7/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!conv3d_7/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    �
conv3d_7/kernel/Regularizer/mulMul*conv3d_7/kernel/Regularizer/mul/x:output:0(conv3d_7/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: {
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_13974*
_output_shapes
:	�$*
dtype0�
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	�$o
dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
dense/kernel/Regularizer/SumSum#dense/kernel/Regularizer/Square:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: c
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    �
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: u
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^conv3d/StatefulPartitionedCall0^conv3d/kernel/Regularizer/Square/ReadVariableOp!^conv3d_1/StatefulPartitionedCall2^conv3d_1/kernel/Regularizer/Square/ReadVariableOp!^conv3d_2/StatefulPartitionedCall2^conv3d_2/kernel/Regularizer/Square/ReadVariableOp!^conv3d_3/StatefulPartitionedCall2^conv3d_3/kernel/Regularizer/Square/ReadVariableOp!^conv3d_4/StatefulPartitionedCall2^conv3d_4/kernel/Regularizer/Square/ReadVariableOp!^conv3d_5/StatefulPartitionedCall2^conv3d_5/kernel/Regularizer/Square/ReadVariableOp!^conv3d_6/StatefulPartitionedCall2^conv3d_6/kernel/Regularizer/Square/ReadVariableOp!^conv3d_7/StatefulPartitionedCall2^conv3d_7/kernel/Regularizer/Square/ReadVariableOp^dense/StatefulPartitionedCall/^dense/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:���������22: : : : : : : : : : : : : : : : : : 2@
conv3d/StatefulPartitionedCallconv3d/StatefulPartitionedCall2b
/conv3d/kernel/Regularizer/Square/ReadVariableOp/conv3d/kernel/Regularizer/Square/ReadVariableOp2D
 conv3d_1/StatefulPartitionedCall conv3d_1/StatefulPartitionedCall2f
1conv3d_1/kernel/Regularizer/Square/ReadVariableOp1conv3d_1/kernel/Regularizer/Square/ReadVariableOp2D
 conv3d_2/StatefulPartitionedCall conv3d_2/StatefulPartitionedCall2f
1conv3d_2/kernel/Regularizer/Square/ReadVariableOp1conv3d_2/kernel/Regularizer/Square/ReadVariableOp2D
 conv3d_3/StatefulPartitionedCall conv3d_3/StatefulPartitionedCall2f
1conv3d_3/kernel/Regularizer/Square/ReadVariableOp1conv3d_3/kernel/Regularizer/Square/ReadVariableOp2D
 conv3d_4/StatefulPartitionedCall conv3d_4/StatefulPartitionedCall2f
1conv3d_4/kernel/Regularizer/Square/ReadVariableOp1conv3d_4/kernel/Regularizer/Square/ReadVariableOp2D
 conv3d_5/StatefulPartitionedCall conv3d_5/StatefulPartitionedCall2f
1conv3d_5/kernel/Regularizer/Square/ReadVariableOp1conv3d_5/kernel/Regularizer/Square/ReadVariableOp2D
 conv3d_6/StatefulPartitionedCall conv3d_6/StatefulPartitionedCall2f
1conv3d_6/kernel/Regularizer/Square/ReadVariableOp1conv3d_6/kernel/Regularizer/Square/ReadVariableOp2D
 conv3d_7/StatefulPartitionedCall conv3d_7/StatefulPartitionedCall2f
1conv3d_7/kernel/Regularizer/Square/ReadVariableOp1conv3d_7/kernel/Regularizer/Square/ReadVariableOp2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2`
.dense/kernel/Regularizer/Square/ReadVariableOp.dense/kernel/Regularizer/Square/ReadVariableOp:\ X
3
_output_shapes!
:���������22
!
_user_specified_name	input_1
�
�
(__inference_conv3d_1_layer_call_fn_16911

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
:���������22@*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *L
fGRE
C__inference_conv3d_1_layer_call_and_return_conditional_losses_13219{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:���������22@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������22@: : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:���������22@
 
_user_specified_nameinputs
�
K
/__inference_max_pooling3d_1_layer_call_fn_16995

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
J__inference_max_pooling3d_1_layer_call_and_return_conditional_losses_13145�
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
�
�
G__inference_add_metric_1_layer_call_and_return_conditional_losses_16876

inputs&
assignaddvariableop_resource: (
assignaddvariableop_1_resource: 

identity_1��AssignAddVariableOp�AssignAddVariableOp_1�div_no_nan/ReadVariableOp�div_no_nan/ReadVariableOp_1F
RankConst*
_output_shapes
: *
dtype0*
value	B : M
range/startConst*
_output_shapes
: *
dtype0*
value	B : M
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :c
rangeRangerange/start:output:0Rank:output:0range/delta:output:0*
_output_shapes
: C
SumSuminputsrange:output:0*
T0*
_output_shapes
: y
AssignAddVariableOpAssignAddVariableOpassignaddvariableop_resourceSum:output:0*
_output_shapes
 *
dtype0F
SizeConst*
_output_shapes
: *
dtype0*
value	B :K
CastCastSize:output:0*

DstT0*

SrcT0*
_output_shapes
: �
AssignAddVariableOp_1AssignAddVariableOpassignaddvariableop_1_resourceCast:y:0^AssignAddVariableOp*
_output_shapes
 *
dtype0�
div_no_nan/ReadVariableOpReadVariableOpassignaddvariableop_resource^AssignAddVariableOp^AssignAddVariableOp_1*
_output_shapes
: *
dtype0�
div_no_nan/ReadVariableOp_1ReadVariableOpassignaddvariableop_1_resource^AssignAddVariableOp_1*
_output_shapes
: *
dtype0

div_no_nanDivNoNan!div_no_nan/ReadVariableOp:value:0#div_no_nan/ReadVariableOp_1:value:0*
T0*
_output_shapes
: E
IdentityIdentitydiv_no_nan:z:0*
T0*
_output_shapes
: F

Identity_1Identityinputs^NoOp*
T0*
_output_shapes
: �
NoOpNoOp^AssignAddVariableOp^AssignAddVariableOp_1^div_no_nan/ReadVariableOp^div_no_nan/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : 2*
AssignAddVariableOpAssignAddVariableOp2.
AssignAddVariableOp_1AssignAddVariableOp_126
div_no_nan/ReadVariableOpdiv_no_nan/ReadVariableOp2:
div_no_nan/ReadVariableOp_1div_no_nan/ReadVariableOp_1:> :

_output_shapes
: 
 
_user_specified_nameinputs
�
�
'__inference_model_1_layer_call_fn_14891
input_2
input_3
input_4
input_5%
unknown:@
	unknown_0:@'
	unknown_1:@@
	unknown_2:@(
	unknown_3:@�
	unknown_4:	�)
	unknown_5:��
	unknown_6:	�)
	unknown_7:��
	unknown_8:	�)
	unknown_9:��

unknown_10:	�*

unknown_11:��

unknown_12:	�*

unknown_13:��

unknown_14:	�

unknown_15:	�$

unknown_16:

unknown_17: 

unknown_18: 

unknown_19

unknown_20

unknown_21

unknown_22: 

unknown_23: 
identity

identity_1��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_2input_3input_4input_5unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23*(
Tin!
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:���������:���������: *4
_read_only_resource_inputs
	
*4
config_proto$"

CPU

GPU(2*0J 8� *K
fFRD
B__inference_model_1_layer_call_and_return_conditional_losses_14774o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������22:���������22:���������:���������: : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
3
_output_shapes!
:���������22
!
_user_specified_name	input_2:\X
3
_output_shapes!
:���������22
!
_user_specified_name	input_3:PL
'
_output_shapes
:���������
!
_user_specified_name	input_4:PL
'
_output_shapes
:���������
!
_user_specified_name	input_5:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
�
@__inference_dense_layer_call_and_return_conditional_losses_17160

inputs1
matmul_readvariableop_resource:	�$-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�.dense/kernel/Regularizer/Square/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�$*
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
:����������
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�$*
dtype0�
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	�$o
dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
dense/kernel/Regularizer/SumSum#dense/kernel/Regularizer/Square:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: c
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    �
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp/^dense/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������$: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2`
.dense/kernel/Regularizer/Square/ReadVariableOp.dense/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:����������$
 
_user_specified_nameinputs
�
f
J__inference_max_pooling3d_1_layer_call_and_return_conditional_losses_13145

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
�
�
@__inference_model_layer_call_and_return_conditional_losses_13738

inputs*
conv3d_13633:@
conv3d_13635:@,
conv3d_1_13638:@@
conv3d_1_13640:@-
conv3d_2_13644:@�
conv3d_2_13646:	�.
conv3d_3_13649:��
conv3d_3_13651:	�.
conv3d_4_13655:��
conv3d_4_13657:	�.
conv3d_5_13660:��
conv3d_5_13662:	�.
conv3d_6_13666:��
conv3d_6_13668:	�.
conv3d_7_13671:��
conv3d_7_13673:	�
dense_13678:	�$
dense_13680:
identity��conv3d/StatefulPartitionedCall�/conv3d/kernel/Regularizer/Square/ReadVariableOp� conv3d_1/StatefulPartitionedCall�1conv3d_1/kernel/Regularizer/Square/ReadVariableOp� conv3d_2/StatefulPartitionedCall�1conv3d_2/kernel/Regularizer/Square/ReadVariableOp� conv3d_3/StatefulPartitionedCall�1conv3d_3/kernel/Regularizer/Square/ReadVariableOp� conv3d_4/StatefulPartitionedCall�1conv3d_4/kernel/Regularizer/Square/ReadVariableOp� conv3d_5/StatefulPartitionedCall�1conv3d_5/kernel/Regularizer/Square/ReadVariableOp� conv3d_6/StatefulPartitionedCall�1conv3d_6/kernel/Regularizer/Square/ReadVariableOp� conv3d_7/StatefulPartitionedCall�1conv3d_7/kernel/Regularizer/Square/ReadVariableOp�dense/StatefulPartitionedCall�.dense/kernel/Regularizer/Square/ReadVariableOp�
conv3d/StatefulPartitionedCallStatefulPartitionedCallinputsconv3d_13633conv3d_13635*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������22@*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *J
fERC
A__inference_conv3d_layer_call_and_return_conditional_losses_13196�
 conv3d_1/StatefulPartitionedCallStatefulPartitionedCall'conv3d/StatefulPartitionedCall:output:0conv3d_1_13638conv3d_1_13640*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������22@*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *L
fGRE
C__inference_conv3d_1_layer_call_and_return_conditional_losses_13219�
max_pooling3d/PartitionedCallPartitionedCall)conv3d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������
@* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8� *Q
fLRJ
H__inference_max_pooling3d_layer_call_and_return_conditional_losses_13133�
 conv3d_2/StatefulPartitionedCallStatefulPartitionedCall&max_pooling3d/PartitionedCall:output:0conv3d_2_13644conv3d_2_13646*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :���������
�*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *L
fGRE
C__inference_conv3d_2_layer_call_and_return_conditional_losses_13243�
 conv3d_3/StatefulPartitionedCallStatefulPartitionedCall)conv3d_2/StatefulPartitionedCall:output:0conv3d_3_13649conv3d_3_13651*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :���������
�*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *L
fGRE
C__inference_conv3d_3_layer_call_and_return_conditional_losses_13266�
max_pooling3d_1/PartitionedCallPartitionedCall)conv3d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :����������* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8� *S
fNRL
J__inference_max_pooling3d_1_layer_call_and_return_conditional_losses_13145�
 conv3d_4/StatefulPartitionedCallStatefulPartitionedCall(max_pooling3d_1/PartitionedCall:output:0conv3d_4_13655conv3d_4_13657*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :����������*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *L
fGRE
C__inference_conv3d_4_layer_call_and_return_conditional_losses_13290�
 conv3d_5/StatefulPartitionedCallStatefulPartitionedCall)conv3d_4/StatefulPartitionedCall:output:0conv3d_5_13660conv3d_5_13662*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :����������*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *L
fGRE
C__inference_conv3d_5_layer_call_and_return_conditional_losses_13313�
max_pooling3d_2/PartitionedCallPartitionedCall)conv3d_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :����������* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8� *S
fNRL
J__inference_max_pooling3d_2_layer_call_and_return_conditional_losses_13157�
 conv3d_6/StatefulPartitionedCallStatefulPartitionedCall(max_pooling3d_2/PartitionedCall:output:0conv3d_6_13666conv3d_6_13668*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :����������*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *L
fGRE
C__inference_conv3d_6_layer_call_and_return_conditional_losses_13337�
 conv3d_7/StatefulPartitionedCallStatefulPartitionedCall)conv3d_6/StatefulPartitionedCall:output:0conv3d_7_13671conv3d_7_13673*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :����������*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *L
fGRE
C__inference_conv3d_7_layer_call_and_return_conditional_losses_13360�
max_pooling3d_3/PartitionedCallPartitionedCall)conv3d_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :����������* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8� *S
fNRL
J__inference_max_pooling3d_3_layer_call_and_return_conditional_losses_13169�
flatten/PartitionedCallPartitionedCall(max_pooling3d_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������$* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8� *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_13373�
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_13678dense_13680*
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
@__inference_dense_layer_call_and_return_conditional_losses_13391�
/conv3d/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv3d_13633**
_output_shapes
:@*
dtype0�
 conv3d/kernel/Regularizer/SquareSquare7conv3d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0**
_output_shapes
:@|
conv3d/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
conv3d/kernel/Regularizer/SumSum$conv3d/kernel/Regularizer/Square:y:0(conv3d/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: d
conv3d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    �
conv3d/kernel/Regularizer/mulMul(conv3d/kernel/Regularizer/mul/x:output:0&conv3d/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1conv3d_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv3d_1_13638**
_output_shapes
:@@*
dtype0�
"conv3d_1/kernel/Regularizer/SquareSquare9conv3d_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0**
_output_shapes
:@@~
!conv3d_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
conv3d_1/kernel/Regularizer/SumSum&conv3d_1/kernel/Regularizer/Square:y:0*conv3d_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!conv3d_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    �
conv3d_1/kernel/Regularizer/mulMul*conv3d_1/kernel/Regularizer/mul/x:output:0(conv3d_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1conv3d_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv3d_2_13644*+
_output_shapes
:@�*
dtype0�
"conv3d_2/kernel/Regularizer/SquareSquare9conv3d_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*+
_output_shapes
:@�~
!conv3d_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
conv3d_2/kernel/Regularizer/SumSum&conv3d_2/kernel/Regularizer/Square:y:0*conv3d_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!conv3d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    �
conv3d_2/kernel/Regularizer/mulMul*conv3d_2/kernel/Regularizer/mul/x:output:0(conv3d_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1conv3d_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv3d_3_13649*,
_output_shapes
:��*
dtype0�
"conv3d_3/kernel/Regularizer/SquareSquare9conv3d_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*,
_output_shapes
:��~
!conv3d_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
conv3d_3/kernel/Regularizer/SumSum&conv3d_3/kernel/Regularizer/Square:y:0*conv3d_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!conv3d_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    �
conv3d_3/kernel/Regularizer/mulMul*conv3d_3/kernel/Regularizer/mul/x:output:0(conv3d_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1conv3d_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv3d_4_13655*,
_output_shapes
:��*
dtype0�
"conv3d_4/kernel/Regularizer/SquareSquare9conv3d_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*,
_output_shapes
:��~
!conv3d_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
conv3d_4/kernel/Regularizer/SumSum&conv3d_4/kernel/Regularizer/Square:y:0*conv3d_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!conv3d_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    �
conv3d_4/kernel/Regularizer/mulMul*conv3d_4/kernel/Regularizer/mul/x:output:0(conv3d_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1conv3d_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv3d_5_13660*,
_output_shapes
:��*
dtype0�
"conv3d_5/kernel/Regularizer/SquareSquare9conv3d_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*,
_output_shapes
:��~
!conv3d_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
conv3d_5/kernel/Regularizer/SumSum&conv3d_5/kernel/Regularizer/Square:y:0*conv3d_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!conv3d_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    �
conv3d_5/kernel/Regularizer/mulMul*conv3d_5/kernel/Regularizer/mul/x:output:0(conv3d_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1conv3d_6/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv3d_6_13666*,
_output_shapes
:��*
dtype0�
"conv3d_6/kernel/Regularizer/SquareSquare9conv3d_6/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*,
_output_shapes
:��~
!conv3d_6/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
conv3d_6/kernel/Regularizer/SumSum&conv3d_6/kernel/Regularizer/Square:y:0*conv3d_6/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!conv3d_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    �
conv3d_6/kernel/Regularizer/mulMul*conv3d_6/kernel/Regularizer/mul/x:output:0(conv3d_6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1conv3d_7/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv3d_7_13671*,
_output_shapes
:��*
dtype0�
"conv3d_7/kernel/Regularizer/SquareSquare9conv3d_7/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*,
_output_shapes
:��~
!conv3d_7/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
conv3d_7/kernel/Regularizer/SumSum&conv3d_7/kernel/Regularizer/Square:y:0*conv3d_7/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!conv3d_7/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    �
conv3d_7/kernel/Regularizer/mulMul*conv3d_7/kernel/Regularizer/mul/x:output:0(conv3d_7/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: {
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_13678*
_output_shapes
:	�$*
dtype0�
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	�$o
dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
dense/kernel/Regularizer/SumSum#dense/kernel/Regularizer/Square:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: c
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    �
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: u
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^conv3d/StatefulPartitionedCall0^conv3d/kernel/Regularizer/Square/ReadVariableOp!^conv3d_1/StatefulPartitionedCall2^conv3d_1/kernel/Regularizer/Square/ReadVariableOp!^conv3d_2/StatefulPartitionedCall2^conv3d_2/kernel/Regularizer/Square/ReadVariableOp!^conv3d_3/StatefulPartitionedCall2^conv3d_3/kernel/Regularizer/Square/ReadVariableOp!^conv3d_4/StatefulPartitionedCall2^conv3d_4/kernel/Regularizer/Square/ReadVariableOp!^conv3d_5/StatefulPartitionedCall2^conv3d_5/kernel/Regularizer/Square/ReadVariableOp!^conv3d_6/StatefulPartitionedCall2^conv3d_6/kernel/Regularizer/Square/ReadVariableOp!^conv3d_7/StatefulPartitionedCall2^conv3d_7/kernel/Regularizer/Square/ReadVariableOp^dense/StatefulPartitionedCall/^dense/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:���������22: : : : : : : : : : : : : : : : : : 2@
conv3d/StatefulPartitionedCallconv3d/StatefulPartitionedCall2b
/conv3d/kernel/Regularizer/Square/ReadVariableOp/conv3d/kernel/Regularizer/Square/ReadVariableOp2D
 conv3d_1/StatefulPartitionedCall conv3d_1/StatefulPartitionedCall2f
1conv3d_1/kernel/Regularizer/Square/ReadVariableOp1conv3d_1/kernel/Regularizer/Square/ReadVariableOp2D
 conv3d_2/StatefulPartitionedCall conv3d_2/StatefulPartitionedCall2f
1conv3d_2/kernel/Regularizer/Square/ReadVariableOp1conv3d_2/kernel/Regularizer/Square/ReadVariableOp2D
 conv3d_3/StatefulPartitionedCall conv3d_3/StatefulPartitionedCall2f
1conv3d_3/kernel/Regularizer/Square/ReadVariableOp1conv3d_3/kernel/Regularizer/Square/ReadVariableOp2D
 conv3d_4/StatefulPartitionedCall conv3d_4/StatefulPartitionedCall2f
1conv3d_4/kernel/Regularizer/Square/ReadVariableOp1conv3d_4/kernel/Regularizer/Square/ReadVariableOp2D
 conv3d_5/StatefulPartitionedCall conv3d_5/StatefulPartitionedCall2f
1conv3d_5/kernel/Regularizer/Square/ReadVariableOp1conv3d_5/kernel/Regularizer/Square/ReadVariableOp2D
 conv3d_6/StatefulPartitionedCall conv3d_6/StatefulPartitionedCall2f
1conv3d_6/kernel/Regularizer/Square/ReadVariableOp1conv3d_6/kernel/Regularizer/Square/ReadVariableOp2D
 conv3d_7/StatefulPartitionedCall conv3d_7/StatefulPartitionedCall2f
1conv3d_7/kernel/Regularizer/Square/ReadVariableOp1conv3d_7/kernel/Regularizer/Square/ReadVariableOp2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2`
.dense/kernel/Regularizer/Square/ReadVariableOp.dense/kernel/Regularizer/Square/ReadVariableOp:[ W
3
_output_shapes!
:���������22
 
_user_specified_nameinputs
��
�)
!__inference__traced_restore_17689
file_prefix<
assignvariableop_conv3d_kernel:@,
assignvariableop_1_conv3d_bias:@@
"assignvariableop_2_conv3d_1_kernel:@@.
 assignvariableop_3_conv3d_1_bias:@A
"assignvariableop_4_conv3d_2_kernel:@�/
 assignvariableop_5_conv3d_2_bias:	�B
"assignvariableop_6_conv3d_3_kernel:��/
 assignvariableop_7_conv3d_3_bias:	�B
"assignvariableop_8_conv3d_4_kernel:��/
 assignvariableop_9_conv3d_4_bias:	�C
#assignvariableop_10_conv3d_5_kernel:��0
!assignvariableop_11_conv3d_5_bias:	�C
#assignvariableop_12_conv3d_6_kernel:��0
!assignvariableop_13_conv3d_6_bias:	�C
#assignvariableop_14_conv3d_7_kernel:��0
!assignvariableop_15_conv3d_7_bias:	�3
 assignvariableop_16_dense_kernel:	�$,
assignvariableop_17_dense_bias:'
assignvariableop_18_adam_iter:	 )
assignvariableop_19_adam_beta_1: )
assignvariableop_20_adam_beta_2: (
assignvariableop_21_adam_decay: 0
&assignvariableop_22_adam_learning_rate: #
assignvariableop_23_total: #
assignvariableop_24_count: .
$assignvariableop_25_add_metric_total: .
$assignvariableop_26_add_metric_count: 0
&assignvariableop_27_add_metric_1_total: 0
&assignvariableop_28_add_metric_1_count: F
(assignvariableop_29_adam_conv3d_kernel_m:@4
&assignvariableop_30_adam_conv3d_bias_m:@H
*assignvariableop_31_adam_conv3d_1_kernel_m:@@6
(assignvariableop_32_adam_conv3d_1_bias_m:@I
*assignvariableop_33_adam_conv3d_2_kernel_m:@�7
(assignvariableop_34_adam_conv3d_2_bias_m:	�J
*assignvariableop_35_adam_conv3d_3_kernel_m:��7
(assignvariableop_36_adam_conv3d_3_bias_m:	�J
*assignvariableop_37_adam_conv3d_4_kernel_m:��7
(assignvariableop_38_adam_conv3d_4_bias_m:	�J
*assignvariableop_39_adam_conv3d_5_kernel_m:��7
(assignvariableop_40_adam_conv3d_5_bias_m:	�J
*assignvariableop_41_adam_conv3d_6_kernel_m:��7
(assignvariableop_42_adam_conv3d_6_bias_m:	�J
*assignvariableop_43_adam_conv3d_7_kernel_m:��7
(assignvariableop_44_adam_conv3d_7_bias_m:	�:
'assignvariableop_45_adam_dense_kernel_m:	�$3
%assignvariableop_46_adam_dense_bias_m:F
(assignvariableop_47_adam_conv3d_kernel_v:@4
&assignvariableop_48_adam_conv3d_bias_v:@H
*assignvariableop_49_adam_conv3d_1_kernel_v:@@6
(assignvariableop_50_adam_conv3d_1_bias_v:@I
*assignvariableop_51_adam_conv3d_2_kernel_v:@�7
(assignvariableop_52_adam_conv3d_2_bias_v:	�J
*assignvariableop_53_adam_conv3d_3_kernel_v:��7
(assignvariableop_54_adam_conv3d_3_bias_v:	�J
*assignvariableop_55_adam_conv3d_4_kernel_v:��7
(assignvariableop_56_adam_conv3d_4_bias_v:	�J
*assignvariableop_57_adam_conv3d_5_kernel_v:��7
(assignvariableop_58_adam_conv3d_5_bias_v:	�J
*assignvariableop_59_adam_conv3d_6_kernel_v:��7
(assignvariableop_60_adam_conv3d_6_bias_v:	�J
*assignvariableop_61_adam_conv3d_7_kernel_v:��7
(assignvariableop_62_adam_conv3d_7_bias_v:	�:
'assignvariableop_63_adam_dense_kernel_v:	�$3
%assignvariableop_64_adam_dense_bias_v:
identity_66��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_36�AssignVariableOp_37�AssignVariableOp_38�AssignVariableOp_39�AssignVariableOp_4�AssignVariableOp_40�AssignVariableOp_41�AssignVariableOp_42�AssignVariableOp_43�AssignVariableOp_44�AssignVariableOp_45�AssignVariableOp_46�AssignVariableOp_47�AssignVariableOp_48�AssignVariableOp_49�AssignVariableOp_5�AssignVariableOp_50�AssignVariableOp_51�AssignVariableOp_52�AssignVariableOp_53�AssignVariableOp_54�AssignVariableOp_55�AssignVariableOp_56�AssignVariableOp_57�AssignVariableOp_58�AssignVariableOp_59�AssignVariableOp_6�AssignVariableOp_60�AssignVariableOp_61�AssignVariableOp_62�AssignVariableOp_63�AssignVariableOp_64�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:B*
dtype0*�
value�B�BB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:B*
dtype0*�
value�B�BB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*P
dtypesF
D2B	[
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
AssignVariableOp_23AssignVariableOpassignvariableop_23_totalIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_24AssignVariableOpassignvariableop_24_countIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_25AssignVariableOp$assignvariableop_25_add_metric_totalIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_26AssignVariableOp$assignvariableop_26_add_metric_countIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_27AssignVariableOp&assignvariableop_27_add_metric_1_totalIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_28AssignVariableOp&assignvariableop_28_add_metric_1_countIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_29AssignVariableOp(assignvariableop_29_adam_conv3d_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_30AssignVariableOp&assignvariableop_30_adam_conv3d_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_31AssignVariableOp*assignvariableop_31_adam_conv3d_1_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_32AssignVariableOp(assignvariableop_32_adam_conv3d_1_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_33AssignVariableOp*assignvariableop_33_adam_conv3d_2_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_34AssignVariableOp(assignvariableop_34_adam_conv3d_2_bias_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_35AssignVariableOp*assignvariableop_35_adam_conv3d_3_kernel_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_36AssignVariableOp(assignvariableop_36_adam_conv3d_3_bias_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_37AssignVariableOp*assignvariableop_37_adam_conv3d_4_kernel_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_38AssignVariableOp(assignvariableop_38_adam_conv3d_4_bias_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_39AssignVariableOp*assignvariableop_39_adam_conv3d_5_kernel_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_40AssignVariableOp(assignvariableop_40_adam_conv3d_5_bias_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_41AssignVariableOp*assignvariableop_41_adam_conv3d_6_kernel_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_42AssignVariableOp(assignvariableop_42_adam_conv3d_6_bias_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_43AssignVariableOp*assignvariableop_43_adam_conv3d_7_kernel_mIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_44AssignVariableOp(assignvariableop_44_adam_conv3d_7_bias_mIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_45AssignVariableOp'assignvariableop_45_adam_dense_kernel_mIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_46AssignVariableOp%assignvariableop_46_adam_dense_bias_mIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_47AssignVariableOp(assignvariableop_47_adam_conv3d_kernel_vIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_48AssignVariableOp&assignvariableop_48_adam_conv3d_bias_vIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_49AssignVariableOp*assignvariableop_49_adam_conv3d_1_kernel_vIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_50AssignVariableOp(assignvariableop_50_adam_conv3d_1_bias_vIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_51AssignVariableOp*assignvariableop_51_adam_conv3d_2_kernel_vIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_52AssignVariableOp(assignvariableop_52_adam_conv3d_2_bias_vIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_53AssignVariableOp*assignvariableop_53_adam_conv3d_3_kernel_vIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_54AssignVariableOp(assignvariableop_54_adam_conv3d_3_bias_vIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_55AssignVariableOp*assignvariableop_55_adam_conv3d_4_kernel_vIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_56AssignVariableOp(assignvariableop_56_adam_conv3d_4_bias_vIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_57AssignVariableOp*assignvariableop_57_adam_conv3d_5_kernel_vIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_58AssignVariableOp(assignvariableop_58_adam_conv3d_5_bias_vIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_59AssignVariableOp*assignvariableop_59_adam_conv3d_6_kernel_vIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_60AssignVariableOp(assignvariableop_60_adam_conv3d_6_bias_vIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_61AssignVariableOp*assignvariableop_61_adam_conv3d_7_kernel_vIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_62AssignVariableOp(assignvariableop_62_adam_conv3d_7_bias_vIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_63AssignVariableOp'assignvariableop_63_adam_dense_kernel_vIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_64AssignVariableOp%assignvariableop_64_adam_dense_bias_vIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 �
Identity_65Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_66IdentityIdentity_65:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_66Identity_66:output:0*�
_input_shapes�
�: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642(
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
__inference_loss_fn_5_17226Z
:conv3d_5_kernel_regularizer_square_readvariableop_resource:��
identity��1conv3d_5/kernel/Regularizer/Square/ReadVariableOp�
1conv3d_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOp:conv3d_5_kernel_regularizer_square_readvariableop_resource*,
_output_shapes
:��*
dtype0�
"conv3d_5/kernel/Regularizer/SquareSquare9conv3d_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*,
_output_shapes
:��~
!conv3d_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
conv3d_5/kernel/Regularizer/SumSum&conv3d_5/kernel/Regularizer/Square:y:0*conv3d_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!conv3d_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    �
conv3d_5/kernel/Regularizer/mulMul*conv3d_5/kernel/Regularizer/mul/x:output:0(conv3d_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: a
IdentityIdentity#conv3d_5/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: z
NoOpNoOp2^conv3d_5/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2f
1conv3d_5/kernel/Regularizer/Square/ReadVariableOp1conv3d_5/kernel/Regularizer/Square/ReadVariableOp
�
�
C__inference_conv3d_6_layer_call_and_return_conditional_losses_13337

inputs>
conv3d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv3D/ReadVariableOp�1conv3d_6/kernel/Regularizer/Square/ReadVariableOp�
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource*,
_output_shapes
:��*
dtype0�
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :����������*
paddingSAME*
strides	
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :����������]
ReluReluBiasAdd:output:0*
T0*4
_output_shapes"
 :�����������
1conv3d_6/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv3d_readvariableop_resource*,
_output_shapes
:��*
dtype0�
"conv3d_6/kernel/Regularizer/SquareSquare9conv3d_6/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*,
_output_shapes
:��~
!conv3d_6/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
conv3d_6/kernel/Regularizer/SumSum&conv3d_6/kernel/Regularizer/Square:y:0*conv3d_6/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!conv3d_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    �
conv3d_6/kernel/Regularizer/mulMul*conv3d_6/kernel/Regularizer/mul/x:output:0(conv3d_6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: n
IdentityIdentityRelu:activations:0^NoOp*
T0*4
_output_shapes"
 :�����������
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp2^conv3d_6/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp2f
1conv3d_6/kernel/Regularizer/Square/ReadVariableOp1conv3d_6/kernel/Regularizer/Square/ReadVariableOp:\ X
4
_output_shapes"
 :����������
 
_user_specified_nameinputs
�
�
C__inference_conv3d_5_layer_call_and_return_conditional_losses_13313

inputs>
conv3d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv3D/ReadVariableOp�1conv3d_5/kernel/Regularizer/Square/ReadVariableOp�
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource*,
_output_shapes
:��*
dtype0�
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :����������*
paddingSAME*
strides	
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :����������]
ReluReluBiasAdd:output:0*
T0*4
_output_shapes"
 :�����������
1conv3d_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv3d_readvariableop_resource*,
_output_shapes
:��*
dtype0�
"conv3d_5/kernel/Regularizer/SquareSquare9conv3d_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*,
_output_shapes
:��~
!conv3d_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
conv3d_5/kernel/Regularizer/SumSum&conv3d_5/kernel/Regularizer/Square:y:0*conv3d_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!conv3d_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    �
conv3d_5/kernel/Regularizer/mulMul*conv3d_5/kernel/Regularizer/mul/x:output:0(conv3d_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: n
IdentityIdentityRelu:activations:0^NoOp*
T0*4
_output_shapes"
 :�����������
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp2^conv3d_5/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp2f
1conv3d_5/kernel/Regularizer/Square/ReadVariableOp1conv3d_5/kernel/Regularizer/Square/ReadVariableOp:\ X
4
_output_shapes"
 :����������
 
_user_specified_nameinputs
�
�
%__inference_model_layer_call_fn_16520

inputs%
unknown:@
	unknown_0:@'
	unknown_1:@@
	unknown_2:@(
	unknown_3:@�
	unknown_4:	�)
	unknown_5:��
	unknown_6:	�)
	unknown_7:��
	unknown_8:	�)
	unknown_9:��

unknown_10:	�*

unknown_11:��

unknown_12:	�*

unknown_13:��

unknown_14:	�

unknown_15:	�$

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
@__inference_model_layer_call_and_return_conditional_losses_13452o
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
�
�
E__inference_add_metric_layer_call_and_return_conditional_losses_16850

inputs&
assignaddvariableop_resource: (
assignaddvariableop_1_resource: 

identity_1��AssignAddVariableOp�AssignAddVariableOp_1�div_no_nan/ReadVariableOp�div_no_nan/ReadVariableOp_1F
RankConst*
_output_shapes
: *
dtype0*
value	B : M
range/startConst*
_output_shapes
: *
dtype0*
value	B : M
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :c
rangeRangerange/start:output:0Rank:output:0range/delta:output:0*
_output_shapes
: C
SumSuminputsrange:output:0*
T0*
_output_shapes
: y
AssignAddVariableOpAssignAddVariableOpassignaddvariableop_resourceSum:output:0*
_output_shapes
 *
dtype0F
SizeConst*
_output_shapes
: *
dtype0*
value	B :K
CastCastSize:output:0*

DstT0*

SrcT0*
_output_shapes
: �
AssignAddVariableOp_1AssignAddVariableOpassignaddvariableop_1_resourceCast:y:0^AssignAddVariableOp*
_output_shapes
 *
dtype0�
div_no_nan/ReadVariableOpReadVariableOpassignaddvariableop_resource^AssignAddVariableOp^AssignAddVariableOp_1*
_output_shapes
: *
dtype0�
div_no_nan/ReadVariableOp_1ReadVariableOpassignaddvariableop_1_resource^AssignAddVariableOp_1*
_output_shapes
: *
dtype0

div_no_nanDivNoNan!div_no_nan/ReadVariableOp:value:0#div_no_nan/ReadVariableOp_1:value:0*
T0*
_output_shapes
: E
IdentityIdentitydiv_no_nan:z:0*
T0*
_output_shapes
: F

Identity_1Identityinputs^NoOp*
T0*
_output_shapes
: �
NoOpNoOp^AssignAddVariableOp^AssignAddVariableOp_1^div_no_nan/ReadVariableOp^div_no_nan/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : 2*
AssignAddVariableOpAssignAddVariableOp2.
AssignAddVariableOp_1AssignAddVariableOp_126
div_no_nan/ReadVariableOpdiv_no_nan/ReadVariableOp2:
div_no_nan/ReadVariableOp_1div_no_nan/ReadVariableOp_1:> :

_output_shapes
: 
 
_user_specified_nameinputs
�
^
B__inference_flatten_layer_call_and_return_conditional_losses_17135

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"����   ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������$Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������$"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������:\ X
4
_output_shapes"
 :����������
 
_user_specified_nameinputs
�
�
__inference_loss_fn_8_17259J
7dense_kernel_regularizer_square_readvariableop_resource:	�$
identity��.dense/kernel/Regularizer/Square/ReadVariableOp�
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOp7dense_kernel_regularizer_square_readvariableop_resource*
_output_shapes
:	�$*
dtype0�
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	�$o
dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
dense/kernel/Regularizer/SumSum#dense/kernel/Regularizer/Square:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: c
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    �
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ^
IdentityIdentity dense/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: w
NoOpNoOp/^dense/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2`
.dense/kernel/Regularizer/Square/ReadVariableOp.dense/kernel/Regularizer/Square/ReadVariableOp
�{
�
__inference__traced_save_17484
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
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop/
+savev2_add_metric_total_read_readvariableop/
+savev2_add_metric_count_read_readvariableop1
-savev2_add_metric_1_total_read_readvariableop1
-savev2_add_metric_1_count_read_readvariableop3
/savev2_adam_conv3d_kernel_m_read_readvariableop1
-savev2_adam_conv3d_bias_m_read_readvariableop5
1savev2_adam_conv3d_1_kernel_m_read_readvariableop3
/savev2_adam_conv3d_1_bias_m_read_readvariableop5
1savev2_adam_conv3d_2_kernel_m_read_readvariableop3
/savev2_adam_conv3d_2_bias_m_read_readvariableop5
1savev2_adam_conv3d_3_kernel_m_read_readvariableop3
/savev2_adam_conv3d_3_bias_m_read_readvariableop5
1savev2_adam_conv3d_4_kernel_m_read_readvariableop3
/savev2_adam_conv3d_4_bias_m_read_readvariableop5
1savev2_adam_conv3d_5_kernel_m_read_readvariableop3
/savev2_adam_conv3d_5_bias_m_read_readvariableop5
1savev2_adam_conv3d_6_kernel_m_read_readvariableop3
/savev2_adam_conv3d_6_bias_m_read_readvariableop5
1savev2_adam_conv3d_7_kernel_m_read_readvariableop3
/savev2_adam_conv3d_7_bias_m_read_readvariableop2
.savev2_adam_dense_kernel_m_read_readvariableop0
,savev2_adam_dense_bias_m_read_readvariableop3
/savev2_adam_conv3d_kernel_v_read_readvariableop1
-savev2_adam_conv3d_bias_v_read_readvariableop5
1savev2_adam_conv3d_1_kernel_v_read_readvariableop3
/savev2_adam_conv3d_1_bias_v_read_readvariableop5
1savev2_adam_conv3d_2_kernel_v_read_readvariableop3
/savev2_adam_conv3d_2_bias_v_read_readvariableop5
1savev2_adam_conv3d_3_kernel_v_read_readvariableop3
/savev2_adam_conv3d_3_bias_v_read_readvariableop5
1savev2_adam_conv3d_4_kernel_v_read_readvariableop3
/savev2_adam_conv3d_4_bias_v_read_readvariableop5
1savev2_adam_conv3d_5_kernel_v_read_readvariableop3
/savev2_adam_conv3d_5_bias_v_read_readvariableop5
1savev2_adam_conv3d_6_kernel_v_read_readvariableop3
/savev2_adam_conv3d_6_bias_v_read_readvariableop5
1savev2_adam_conv3d_7_kernel_v_read_readvariableop3
/savev2_adam_conv3d_7_bias_v_read_readvariableop2
.savev2_adam_dense_kernel_v_read_readvariableop0
,savev2_adam_dense_bias_v_read_readvariableop
savev2_const_3

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
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:B*
dtype0*�
value�B�BB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:B*
dtype0*�
value�B�BB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0(savev2_conv3d_kernel_read_readvariableop&savev2_conv3d_bias_read_readvariableop*savev2_conv3d_1_kernel_read_readvariableop(savev2_conv3d_1_bias_read_readvariableop*savev2_conv3d_2_kernel_read_readvariableop(savev2_conv3d_2_bias_read_readvariableop*savev2_conv3d_3_kernel_read_readvariableop(savev2_conv3d_3_bias_read_readvariableop*savev2_conv3d_4_kernel_read_readvariableop(savev2_conv3d_4_bias_read_readvariableop*savev2_conv3d_5_kernel_read_readvariableop(savev2_conv3d_5_bias_read_readvariableop*savev2_conv3d_6_kernel_read_readvariableop(savev2_conv3d_6_bias_read_readvariableop*savev2_conv3d_7_kernel_read_readvariableop(savev2_conv3d_7_bias_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop+savev2_add_metric_total_read_readvariableop+savev2_add_metric_count_read_readvariableop-savev2_add_metric_1_total_read_readvariableop-savev2_add_metric_1_count_read_readvariableop/savev2_adam_conv3d_kernel_m_read_readvariableop-savev2_adam_conv3d_bias_m_read_readvariableop1savev2_adam_conv3d_1_kernel_m_read_readvariableop/savev2_adam_conv3d_1_bias_m_read_readvariableop1savev2_adam_conv3d_2_kernel_m_read_readvariableop/savev2_adam_conv3d_2_bias_m_read_readvariableop1savev2_adam_conv3d_3_kernel_m_read_readvariableop/savev2_adam_conv3d_3_bias_m_read_readvariableop1savev2_adam_conv3d_4_kernel_m_read_readvariableop/savev2_adam_conv3d_4_bias_m_read_readvariableop1savev2_adam_conv3d_5_kernel_m_read_readvariableop/savev2_adam_conv3d_5_bias_m_read_readvariableop1savev2_adam_conv3d_6_kernel_m_read_readvariableop/savev2_adam_conv3d_6_bias_m_read_readvariableop1savev2_adam_conv3d_7_kernel_m_read_readvariableop/savev2_adam_conv3d_7_bias_m_read_readvariableop.savev2_adam_dense_kernel_m_read_readvariableop,savev2_adam_dense_bias_m_read_readvariableop/savev2_adam_conv3d_kernel_v_read_readvariableop-savev2_adam_conv3d_bias_v_read_readvariableop1savev2_adam_conv3d_1_kernel_v_read_readvariableop/savev2_adam_conv3d_1_bias_v_read_readvariableop1savev2_adam_conv3d_2_kernel_v_read_readvariableop/savev2_adam_conv3d_2_bias_v_read_readvariableop1savev2_adam_conv3d_3_kernel_v_read_readvariableop/savev2_adam_conv3d_3_bias_v_read_readvariableop1savev2_adam_conv3d_4_kernel_v_read_readvariableop/savev2_adam_conv3d_4_bias_v_read_readvariableop1savev2_adam_conv3d_5_kernel_v_read_readvariableop/savev2_adam_conv3d_5_bias_v_read_readvariableop1savev2_adam_conv3d_6_kernel_v_read_readvariableop/savev2_adam_conv3d_6_bias_v_read_readvariableop1savev2_adam_conv3d_7_kernel_v_read_readvariableop/savev2_adam_conv3d_7_bias_v_read_readvariableop.savev2_adam_dense_kernel_v_read_readvariableop,savev2_adam_dense_bias_v_read_readvariableopsavev2_const_3"/device:CPU:0*
_output_shapes
 *P
dtypesF
D2B	�
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

identity_1Identity_1:output:0*�
_input_shapes�
�: :@:@:@@:@:@�:�:��:�:��:�:��:�:��:�:��:�:	�$:: : : : : : : : : : : :@:@:@@:@:@�:�:��:�:��:�:��:�:��:�:��:�:	�$::@:@:@@:@:@�:�:��:�:��:�:��:�:��:�:��:�:	�$:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:0,
*
_output_shapes
:@: 

_output_shapes
:@:0,
*
_output_shapes
:@@: 

_output_shapes
:@:1-
+
_output_shapes
:@�:!

_output_shapes	
:�:2.
,
_output_shapes
:��:!

_output_shapes	
:�:2	.
,
_output_shapes
:��:!


_output_shapes	
:�:2.
,
_output_shapes
:��:!

_output_shapes	
:�:2.
,
_output_shapes
:��:!

_output_shapes	
:�:2.
,
_output_shapes
:��:!

_output_shapes	
:�:%!

_output_shapes
:	�$: 
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
: :

_output_shapes
: :

_output_shapes
: :0,
*
_output_shapes
:@: 

_output_shapes
:@:0 ,
*
_output_shapes
:@@: !

_output_shapes
:@:1"-
+
_output_shapes
:@�:!#

_output_shapes	
:�:2$.
,
_output_shapes
:��:!%

_output_shapes	
:�:2&.
,
_output_shapes
:��:!'

_output_shapes	
:�:2(.
,
_output_shapes
:��:!)

_output_shapes	
:�:2*.
,
_output_shapes
:��:!+

_output_shapes	
:�:2,.
,
_output_shapes
:��:!-

_output_shapes	
:�:%.!

_output_shapes
:	�$: /

_output_shapes
::00,
*
_output_shapes
:@: 1

_output_shapes
:@:02,
*
_output_shapes
:@@: 3

_output_shapes
:@:14-
+
_output_shapes
:@�:!5

_output_shapes	
:�:26.
,
_output_shapes
:��:!7

_output_shapes	
:�:28.
,
_output_shapes
:��:!9

_output_shapes	
:�:2:.
,
_output_shapes
:��:!;

_output_shapes	
:�:2<.
,
_output_shapes
:��:!=

_output_shapes	
:�:2>.
,
_output_shapes
:��:!?

_output_shapes	
:�:%@!

_output_shapes
:	�$: A

_output_shapes
::B

_output_shapes
: 
�
�
,__inference_add_metric_1_layer_call_fn_16859

inputs
unknown: 
	unknown_0: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
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
GPU(2*0J 8� *P
fKRI
G__inference_add_metric_1_layer_call_and_return_conditional_losses_14291^
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes
: `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : 22
StatefulPartitionedCallStatefulPartitionedCall:> :

_output_shapes
: 
 
_user_specified_nameinputs
��
�
B__inference_model_1_layer_call_and_return_conditional_losses_16078
inputs_0
inputs_1
inputs_2
inputs_3I
+model_conv3d_conv3d_readvariableop_resource:@:
,model_conv3d_biasadd_readvariableop_resource:@K
-model_conv3d_1_conv3d_readvariableop_resource:@@<
.model_conv3d_1_biasadd_readvariableop_resource:@L
-model_conv3d_2_conv3d_readvariableop_resource:@�=
.model_conv3d_2_biasadd_readvariableop_resource:	�M
-model_conv3d_3_conv3d_readvariableop_resource:��=
.model_conv3d_3_biasadd_readvariableop_resource:	�M
-model_conv3d_4_conv3d_readvariableop_resource:��=
.model_conv3d_4_biasadd_readvariableop_resource:	�M
-model_conv3d_5_conv3d_readvariableop_resource:��=
.model_conv3d_5_biasadd_readvariableop_resource:	�M
-model_conv3d_6_conv3d_readvariableop_resource:��=
.model_conv3d_6_biasadd_readvariableop_resource:	�M
-model_conv3d_7_conv3d_readvariableop_resource:��=
.model_conv3d_7_biasadd_readvariableop_resource:	�=
*model_dense_matmul_readvariableop_resource:	�$9
+model_dense_biasadd_readvariableop_resource:1
'add_metric_assignaddvariableop_resource: 3
)add_metric_assignaddvariableop_1_resource: 
unknown
	unknown_0
	unknown_13
)add_metric_1_assignaddvariableop_resource: 5
+add_metric_1_assignaddvariableop_1_resource: 
identity

identity_1

identity_2��add_metric/AssignAddVariableOp� add_metric/AssignAddVariableOp_1�$add_metric/div_no_nan/ReadVariableOp�&add_metric/div_no_nan/ReadVariableOp_1� add_metric_1/AssignAddVariableOp�"add_metric_1/AssignAddVariableOp_1�&add_metric_1/div_no_nan/ReadVariableOp�(add_metric_1/div_no_nan/ReadVariableOp_1�/conv3d/kernel/Regularizer/Square/ReadVariableOp�1conv3d_1/kernel/Regularizer/Square/ReadVariableOp�1conv3d_2/kernel/Regularizer/Square/ReadVariableOp�1conv3d_3/kernel/Regularizer/Square/ReadVariableOp�1conv3d_4/kernel/Regularizer/Square/ReadVariableOp�1conv3d_5/kernel/Regularizer/Square/ReadVariableOp�1conv3d_6/kernel/Regularizer/Square/ReadVariableOp�1conv3d_7/kernel/Regularizer/Square/ReadVariableOp�.dense/kernel/Regularizer/Square/ReadVariableOp�#model/conv3d/BiasAdd/ReadVariableOp�%model/conv3d/BiasAdd_1/ReadVariableOp�"model/conv3d/Conv3D/ReadVariableOp�$model/conv3d/Conv3D_1/ReadVariableOp�%model/conv3d_1/BiasAdd/ReadVariableOp�'model/conv3d_1/BiasAdd_1/ReadVariableOp�$model/conv3d_1/Conv3D/ReadVariableOp�&model/conv3d_1/Conv3D_1/ReadVariableOp�%model/conv3d_2/BiasAdd/ReadVariableOp�'model/conv3d_2/BiasAdd_1/ReadVariableOp�$model/conv3d_2/Conv3D/ReadVariableOp�&model/conv3d_2/Conv3D_1/ReadVariableOp�%model/conv3d_3/BiasAdd/ReadVariableOp�'model/conv3d_3/BiasAdd_1/ReadVariableOp�$model/conv3d_3/Conv3D/ReadVariableOp�&model/conv3d_3/Conv3D_1/ReadVariableOp�%model/conv3d_4/BiasAdd/ReadVariableOp�'model/conv3d_4/BiasAdd_1/ReadVariableOp�$model/conv3d_4/Conv3D/ReadVariableOp�&model/conv3d_4/Conv3D_1/ReadVariableOp�%model/conv3d_5/BiasAdd/ReadVariableOp�'model/conv3d_5/BiasAdd_1/ReadVariableOp�$model/conv3d_5/Conv3D/ReadVariableOp�&model/conv3d_5/Conv3D_1/ReadVariableOp�%model/conv3d_6/BiasAdd/ReadVariableOp�'model/conv3d_6/BiasAdd_1/ReadVariableOp�$model/conv3d_6/Conv3D/ReadVariableOp�&model/conv3d_6/Conv3D_1/ReadVariableOp�%model/conv3d_7/BiasAdd/ReadVariableOp�'model/conv3d_7/BiasAdd_1/ReadVariableOp�$model/conv3d_7/Conv3D/ReadVariableOp�&model/conv3d_7/Conv3D_1/ReadVariableOp�"model/dense/BiasAdd/ReadVariableOp�$model/dense/BiasAdd_1/ReadVariableOp�!model/dense/MatMul/ReadVariableOp�#model/dense/MatMul_1/ReadVariableOp�
"model/conv3d/Conv3D/ReadVariableOpReadVariableOp+model_conv3d_conv3d_readvariableop_resource**
_output_shapes
:@*
dtype0�
model/conv3d/Conv3DConv3Dinputs_1*model/conv3d/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������22@*
paddingSAME*
strides	
�
#model/conv3d/BiasAdd/ReadVariableOpReadVariableOp,model_conv3d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
model/conv3d/BiasAddBiasAddmodel/conv3d/Conv3D:output:0+model/conv3d/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������22@v
model/conv3d/ReluRelumodel/conv3d/BiasAdd:output:0*
T0*3
_output_shapes!
:���������22@�
$model/conv3d_1/Conv3D/ReadVariableOpReadVariableOp-model_conv3d_1_conv3d_readvariableop_resource**
_output_shapes
:@@*
dtype0�
model/conv3d_1/Conv3DConv3Dmodel/conv3d/Relu:activations:0,model/conv3d_1/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������22@*
paddingSAME*
strides	
�
%model/conv3d_1/BiasAdd/ReadVariableOpReadVariableOp.model_conv3d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
model/conv3d_1/BiasAddBiasAddmodel/conv3d_1/Conv3D:output:0-model/conv3d_1/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������22@z
model/conv3d_1/ReluRelumodel/conv3d_1/BiasAdd:output:0*
T0*3
_output_shapes!
:���������22@�
model/max_pooling3d/MaxPool3D	MaxPool3D!model/conv3d_1/Relu:activations:0*
T0*3
_output_shapes!
:���������
@*
ksize	
*
paddingVALID*
strides	
�
$model/conv3d_2/Conv3D/ReadVariableOpReadVariableOp-model_conv3d_2_conv3d_readvariableop_resource*+
_output_shapes
:@�*
dtype0�
model/conv3d_2/Conv3DConv3D&model/max_pooling3d/MaxPool3D:output:0,model/conv3d_2/Conv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :���������
�*
paddingSAME*
strides	
�
%model/conv3d_2/BiasAdd/ReadVariableOpReadVariableOp.model_conv3d_2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
model/conv3d_2/BiasAddBiasAddmodel/conv3d_2/Conv3D:output:0-model/conv3d_2/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :���������
�{
model/conv3d_2/ReluRelumodel/conv3d_2/BiasAdd:output:0*
T0*4
_output_shapes"
 :���������
��
$model/conv3d_3/Conv3D/ReadVariableOpReadVariableOp-model_conv3d_3_conv3d_readvariableop_resource*,
_output_shapes
:��*
dtype0�
model/conv3d_3/Conv3DConv3D!model/conv3d_2/Relu:activations:0,model/conv3d_3/Conv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :���������
�*
paddingSAME*
strides	
�
%model/conv3d_3/BiasAdd/ReadVariableOpReadVariableOp.model_conv3d_3_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
model/conv3d_3/BiasAddBiasAddmodel/conv3d_3/Conv3D:output:0-model/conv3d_3/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :���������
�{
model/conv3d_3/ReluRelumodel/conv3d_3/BiasAdd:output:0*
T0*4
_output_shapes"
 :���������
��
model/max_pooling3d_1/MaxPool3D	MaxPool3D!model/conv3d_3/Relu:activations:0*
T0*4
_output_shapes"
 :����������*
ksize	
*
paddingVALID*
strides	
�
$model/conv3d_4/Conv3D/ReadVariableOpReadVariableOp-model_conv3d_4_conv3d_readvariableop_resource*,
_output_shapes
:��*
dtype0�
model/conv3d_4/Conv3DConv3D(model/max_pooling3d_1/MaxPool3D:output:0,model/conv3d_4/Conv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :����������*
paddingSAME*
strides	
�
%model/conv3d_4/BiasAdd/ReadVariableOpReadVariableOp.model_conv3d_4_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
model/conv3d_4/BiasAddBiasAddmodel/conv3d_4/Conv3D:output:0-model/conv3d_4/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :����������{
model/conv3d_4/ReluRelumodel/conv3d_4/BiasAdd:output:0*
T0*4
_output_shapes"
 :�����������
$model/conv3d_5/Conv3D/ReadVariableOpReadVariableOp-model_conv3d_5_conv3d_readvariableop_resource*,
_output_shapes
:��*
dtype0�
model/conv3d_5/Conv3DConv3D!model/conv3d_4/Relu:activations:0,model/conv3d_5/Conv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :����������*
paddingSAME*
strides	
�
%model/conv3d_5/BiasAdd/ReadVariableOpReadVariableOp.model_conv3d_5_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
model/conv3d_5/BiasAddBiasAddmodel/conv3d_5/Conv3D:output:0-model/conv3d_5/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :����������{
model/conv3d_5/ReluRelumodel/conv3d_5/BiasAdd:output:0*
T0*4
_output_shapes"
 :�����������
model/max_pooling3d_2/MaxPool3D	MaxPool3D!model/conv3d_5/Relu:activations:0*
T0*4
_output_shapes"
 :����������*
ksize	
*
paddingVALID*
strides	
�
$model/conv3d_6/Conv3D/ReadVariableOpReadVariableOp-model_conv3d_6_conv3d_readvariableop_resource*,
_output_shapes
:��*
dtype0�
model/conv3d_6/Conv3DConv3D(model/max_pooling3d_2/MaxPool3D:output:0,model/conv3d_6/Conv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :����������*
paddingSAME*
strides	
�
%model/conv3d_6/BiasAdd/ReadVariableOpReadVariableOp.model_conv3d_6_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
model/conv3d_6/BiasAddBiasAddmodel/conv3d_6/Conv3D:output:0-model/conv3d_6/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :����������{
model/conv3d_6/ReluRelumodel/conv3d_6/BiasAdd:output:0*
T0*4
_output_shapes"
 :�����������
$model/conv3d_7/Conv3D/ReadVariableOpReadVariableOp-model_conv3d_7_conv3d_readvariableop_resource*,
_output_shapes
:��*
dtype0�
model/conv3d_7/Conv3DConv3D!model/conv3d_6/Relu:activations:0,model/conv3d_7/Conv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :����������*
paddingSAME*
strides	
�
%model/conv3d_7/BiasAdd/ReadVariableOpReadVariableOp.model_conv3d_7_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
model/conv3d_7/BiasAddBiasAddmodel/conv3d_7/Conv3D:output:0-model/conv3d_7/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :����������{
model/conv3d_7/ReluRelumodel/conv3d_7/BiasAdd:output:0*
T0*4
_output_shapes"
 :�����������
model/max_pooling3d_3/MaxPool3D	MaxPool3D!model/conv3d_7/Relu:activations:0*
T0*4
_output_shapes"
 :����������*
ksize	
*
paddingVALID*
strides	
d
model/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   �
model/flatten/ReshapeReshape(model/max_pooling3d_3/MaxPool3D:output:0model/flatten/Const:output:0*
T0*(
_output_shapes
:����������$�
!model/dense/MatMul/ReadVariableOpReadVariableOp*model_dense_matmul_readvariableop_resource*
_output_shapes
:	�$*
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
:����������
$model/conv3d/Conv3D_1/ReadVariableOpReadVariableOp+model_conv3d_conv3d_readvariableop_resource**
_output_shapes
:@*
dtype0�
model/conv3d/Conv3D_1Conv3Dinputs_0,model/conv3d/Conv3D_1/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������22@*
paddingSAME*
strides	
�
%model/conv3d/BiasAdd_1/ReadVariableOpReadVariableOp,model_conv3d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
model/conv3d/BiasAdd_1BiasAddmodel/conv3d/Conv3D_1:output:0-model/conv3d/BiasAdd_1/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������22@z
model/conv3d/Relu_1Relumodel/conv3d/BiasAdd_1:output:0*
T0*3
_output_shapes!
:���������22@�
&model/conv3d_1/Conv3D_1/ReadVariableOpReadVariableOp-model_conv3d_1_conv3d_readvariableop_resource**
_output_shapes
:@@*
dtype0�
model/conv3d_1/Conv3D_1Conv3D!model/conv3d/Relu_1:activations:0.model/conv3d_1/Conv3D_1/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������22@*
paddingSAME*
strides	
�
'model/conv3d_1/BiasAdd_1/ReadVariableOpReadVariableOp.model_conv3d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
model/conv3d_1/BiasAdd_1BiasAdd model/conv3d_1/Conv3D_1:output:0/model/conv3d_1/BiasAdd_1/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������22@~
model/conv3d_1/Relu_1Relu!model/conv3d_1/BiasAdd_1:output:0*
T0*3
_output_shapes!
:���������22@�
model/max_pooling3d/MaxPool3D_1	MaxPool3D#model/conv3d_1/Relu_1:activations:0*
T0*3
_output_shapes!
:���������
@*
ksize	
*
paddingVALID*
strides	
�
&model/conv3d_2/Conv3D_1/ReadVariableOpReadVariableOp-model_conv3d_2_conv3d_readvariableop_resource*+
_output_shapes
:@�*
dtype0�
model/conv3d_2/Conv3D_1Conv3D(model/max_pooling3d/MaxPool3D_1:output:0.model/conv3d_2/Conv3D_1/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :���������
�*
paddingSAME*
strides	
�
'model/conv3d_2/BiasAdd_1/ReadVariableOpReadVariableOp.model_conv3d_2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
model/conv3d_2/BiasAdd_1BiasAdd model/conv3d_2/Conv3D_1:output:0/model/conv3d_2/BiasAdd_1/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :���������
�
model/conv3d_2/Relu_1Relu!model/conv3d_2/BiasAdd_1:output:0*
T0*4
_output_shapes"
 :���������
��
&model/conv3d_3/Conv3D_1/ReadVariableOpReadVariableOp-model_conv3d_3_conv3d_readvariableop_resource*,
_output_shapes
:��*
dtype0�
model/conv3d_3/Conv3D_1Conv3D#model/conv3d_2/Relu_1:activations:0.model/conv3d_3/Conv3D_1/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :���������
�*
paddingSAME*
strides	
�
'model/conv3d_3/BiasAdd_1/ReadVariableOpReadVariableOp.model_conv3d_3_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
model/conv3d_3/BiasAdd_1BiasAdd model/conv3d_3/Conv3D_1:output:0/model/conv3d_3/BiasAdd_1/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :���������
�
model/conv3d_3/Relu_1Relu!model/conv3d_3/BiasAdd_1:output:0*
T0*4
_output_shapes"
 :���������
��
!model/max_pooling3d_1/MaxPool3D_1	MaxPool3D#model/conv3d_3/Relu_1:activations:0*
T0*4
_output_shapes"
 :����������*
ksize	
*
paddingVALID*
strides	
�
&model/conv3d_4/Conv3D_1/ReadVariableOpReadVariableOp-model_conv3d_4_conv3d_readvariableop_resource*,
_output_shapes
:��*
dtype0�
model/conv3d_4/Conv3D_1Conv3D*model/max_pooling3d_1/MaxPool3D_1:output:0.model/conv3d_4/Conv3D_1/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :����������*
paddingSAME*
strides	
�
'model/conv3d_4/BiasAdd_1/ReadVariableOpReadVariableOp.model_conv3d_4_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
model/conv3d_4/BiasAdd_1BiasAdd model/conv3d_4/Conv3D_1:output:0/model/conv3d_4/BiasAdd_1/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :����������
model/conv3d_4/Relu_1Relu!model/conv3d_4/BiasAdd_1:output:0*
T0*4
_output_shapes"
 :�����������
&model/conv3d_5/Conv3D_1/ReadVariableOpReadVariableOp-model_conv3d_5_conv3d_readvariableop_resource*,
_output_shapes
:��*
dtype0�
model/conv3d_5/Conv3D_1Conv3D#model/conv3d_4/Relu_1:activations:0.model/conv3d_5/Conv3D_1/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :����������*
paddingSAME*
strides	
�
'model/conv3d_5/BiasAdd_1/ReadVariableOpReadVariableOp.model_conv3d_5_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
model/conv3d_5/BiasAdd_1BiasAdd model/conv3d_5/Conv3D_1:output:0/model/conv3d_5/BiasAdd_1/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :����������
model/conv3d_5/Relu_1Relu!model/conv3d_5/BiasAdd_1:output:0*
T0*4
_output_shapes"
 :�����������
!model/max_pooling3d_2/MaxPool3D_1	MaxPool3D#model/conv3d_5/Relu_1:activations:0*
T0*4
_output_shapes"
 :����������*
ksize	
*
paddingVALID*
strides	
�
&model/conv3d_6/Conv3D_1/ReadVariableOpReadVariableOp-model_conv3d_6_conv3d_readvariableop_resource*,
_output_shapes
:��*
dtype0�
model/conv3d_6/Conv3D_1Conv3D*model/max_pooling3d_2/MaxPool3D_1:output:0.model/conv3d_6/Conv3D_1/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :����������*
paddingSAME*
strides	
�
'model/conv3d_6/BiasAdd_1/ReadVariableOpReadVariableOp.model_conv3d_6_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
model/conv3d_6/BiasAdd_1BiasAdd model/conv3d_6/Conv3D_1:output:0/model/conv3d_6/BiasAdd_1/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :����������
model/conv3d_6/Relu_1Relu!model/conv3d_6/BiasAdd_1:output:0*
T0*4
_output_shapes"
 :�����������
&model/conv3d_7/Conv3D_1/ReadVariableOpReadVariableOp-model_conv3d_7_conv3d_readvariableop_resource*,
_output_shapes
:��*
dtype0�
model/conv3d_7/Conv3D_1Conv3D#model/conv3d_6/Relu_1:activations:0.model/conv3d_7/Conv3D_1/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :����������*
paddingSAME*
strides	
�
'model/conv3d_7/BiasAdd_1/ReadVariableOpReadVariableOp.model_conv3d_7_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
model/conv3d_7/BiasAdd_1BiasAdd model/conv3d_7/Conv3D_1:output:0/model/conv3d_7/BiasAdd_1/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :����������
model/conv3d_7/Relu_1Relu!model/conv3d_7/BiasAdd_1:output:0*
T0*4
_output_shapes"
 :�����������
!model/max_pooling3d_3/MaxPool3D_1	MaxPool3D#model/conv3d_7/Relu_1:activations:0*
T0*4
_output_shapes"
 :����������*
ksize	
*
paddingVALID*
strides	
f
model/flatten/Const_1Const*
_output_shapes
:*
dtype0*
valueB"����   �
model/flatten/Reshape_1Reshape*model/max_pooling3d_3/MaxPool3D_1:output:0model/flatten/Const_1:output:0*
T0*(
_output_shapes
:����������$�
#model/dense/MatMul_1/ReadVariableOpReadVariableOp*model_dense_matmul_readvariableop_resource*
_output_shapes
:	�$*
dtype0�
model/dense/MatMul_1MatMul model/flatten/Reshape_1:output:0+model/dense/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
$model/dense/BiasAdd_1/ReadVariableOpReadVariableOp+model_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model/dense/BiasAdd_1BiasAddmodel/dense/MatMul_1:product:0,model/dense/BiasAdd_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������V
tf.compat.v1.squeeze_1/SqueezeSqueezeinputs_3*
T0*
_output_shapes
:j
tf.compat.v1.squeeze_3/SqueezeSqueezemodel/dense/BiasAdd:output:0*
T0*
_output_shapes
:T
tf.compat.v1.squeeze/SqueezeSqueezeinputs_2*
T0*
_output_shapes
:l
tf.compat.v1.squeeze_2/SqueezeSqueezemodel/dense/BiasAdd_1:output:0*
T0*
_output_shapes
:l
tf.compat.v1.squeeze_4/SqueezeSqueezemodel/dense/BiasAdd_1:output:0*
T0*
_output_shapes
:j
tf.compat.v1.squeeze_5/SqueezeSqueezemodel/dense/BiasAdd:output:0*
T0*
_output_shapes
:V
tf.compat.v1.squeeze_7/SqueezeSqueezeinputs_3*
T0*
_output_shapes
:j
tf.compat.v1.squeeze_9/SqueezeSqueezemodel/dense/BiasAdd:output:0*
T0*
_output_shapes
:V
tf.compat.v1.squeeze_6/SqueezeSqueezeinputs_2*
T0*
_output_shapes
:l
tf.compat.v1.squeeze_8/SqueezeSqueezemodel/dense/BiasAdd_1:output:0*
T0*
_output_shapes
:k
tf.compat.v1.squeeze_13/SqueezeSqueezemodel/dense/BiasAdd:output:0*
T0*
_output_shapes
:W
tf.compat.v1.squeeze_12/SqueezeSqueezeinputs_3*
T0*
_output_shapes
:m
tf.compat.v1.squeeze_11/SqueezeSqueezemodel/dense/BiasAdd_1:output:0*
T0*
_output_shapes
:W
tf.compat.v1.squeeze_10/SqueezeSqueezeinputs_2*
T0*
_output_shapes
:�
tf.math.subtract_1/SubSub'tf.compat.v1.squeeze_1/Squeeze:output:0'tf.compat.v1.squeeze_3/Squeeze:output:0*
T0*
_output_shapes
:�
tf.math.subtract/SubSub%tf.compat.v1.squeeze/Squeeze:output:0'tf.compat.v1.squeeze_2/Squeeze:output:0*
T0*
_output_shapes
:�
tf.math.subtract_2/SubSub'tf.compat.v1.squeeze_4/Squeeze:output:0'tf.compat.v1.squeeze_5/Squeeze:output:0*
T0*
_output_shapes
:�
tf.math.subtract_5/SubSub'tf.compat.v1.squeeze_7/Squeeze:output:0'tf.compat.v1.squeeze_9/Squeeze:output:0*
T0*
_output_shapes
:�
tf.math.subtract_4/SubSub'tf.compat.v1.squeeze_6/Squeeze:output:0'tf.compat.v1.squeeze_8/Squeeze:output:0*
T0*
_output_shapes
:m
tf.math.reduce_mean_8/RankRank(tf.compat.v1.squeeze_13/Squeeze:output:0*
T0*
_output_shapes
: c
!tf.math.reduce_mean_8/range/startConst*
_output_shapes
: *
dtype0*
value	B : c
!tf.math.reduce_mean_8/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
tf.math.reduce_mean_8/rangeRange*tf.math.reduce_mean_8/range/start:output:0#tf.math.reduce_mean_8/Rank:output:0*tf.math.reduce_mean_8/range/delta:output:0*#
_output_shapes
:����������
tf.math.reduce_mean_8/MeanMean(tf.compat.v1.squeeze_13/Squeeze:output:0$tf.math.reduce_mean_8/range:output:0*
T0*
_output_shapes
: m
tf.math.reduce_mean_7/RankRank(tf.compat.v1.squeeze_12/Squeeze:output:0*
T0*
_output_shapes
: c
!tf.math.reduce_mean_7/range/startConst*
_output_shapes
: *
dtype0*
value	B : c
!tf.math.reduce_mean_7/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
tf.math.reduce_mean_7/rangeRange*tf.math.reduce_mean_7/range/start:output:0#tf.math.reduce_mean_7/Rank:output:0*tf.math.reduce_mean_7/range/delta:output:0*#
_output_shapes
:����������
tf.math.reduce_mean_7/MeanMean(tf.compat.v1.squeeze_12/Squeeze:output:0$tf.math.reduce_mean_7/range:output:0*
T0*
_output_shapes
: m
tf.math.reduce_mean_6/RankRank(tf.compat.v1.squeeze_11/Squeeze:output:0*
T0*
_output_shapes
: c
!tf.math.reduce_mean_6/range/startConst*
_output_shapes
: *
dtype0*
value	B : c
!tf.math.reduce_mean_6/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
tf.math.reduce_mean_6/rangeRange*tf.math.reduce_mean_6/range/start:output:0#tf.math.reduce_mean_6/Rank:output:0*tf.math.reduce_mean_6/range/delta:output:0*#
_output_shapes
:����������
tf.math.reduce_mean_6/MeanMean(tf.compat.v1.squeeze_11/Squeeze:output:0$tf.math.reduce_mean_6/range:output:0*
T0*
_output_shapes
: m
tf.math.reduce_mean_5/RankRank(tf.compat.v1.squeeze_10/Squeeze:output:0*
T0*
_output_shapes
: c
!tf.math.reduce_mean_5/range/startConst*
_output_shapes
: *
dtype0*
value	B : c
!tf.math.reduce_mean_5/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
tf.math.reduce_mean_5/rangeRange*tf.math.reduce_mean_5/range/start:output:0#tf.math.reduce_mean_5/Rank:output:0*tf.math.reduce_mean_5/range/delta:output:0*#
_output_shapes
:����������
tf.math.reduce_mean_5/MeanMean(tf.compat.v1.squeeze_10/Squeeze:output:0$tf.math.reduce_mean_5/range:output:0*
T0*
_output_shapes
: `
tf.math.square_1/SquareSquaretf.math.subtract_1/Sub:z:0*
T0*
_output_shapes
:\
tf.math.square/SquareSquaretf.math.subtract/Sub:z:0*
T0*
_output_shapes
:`
tf.math.square_2/SquareSquaretf.math.subtract_2/Sub:z:0*
T0*
_output_shapes
:`
tf.math.square_4/SquareSquaretf.math.subtract_5/Sub:z:0*
T0*
_output_shapes
:`
tf.math.square_3/SquareSquaretf.math.subtract_4/Sub:z:0*
T0*
_output_shapes
:�
tf.math.subtract_10/SubSub(tf.compat.v1.squeeze_13/Squeeze:output:0#tf.math.reduce_mean_8/Mean:output:0*
T0*
_output_shapes
:�
tf.math.subtract_9/SubSub(tf.compat.v1.squeeze_12/Squeeze:output:0#tf.math.reduce_mean_7/Mean:output:0*
T0*
_output_shapes
:�
tf.math.subtract_7/SubSub(tf.compat.v1.squeeze_11/Squeeze:output:0#tf.math.reduce_mean_6/Mean:output:0*
T0*
_output_shapes
:�
tf.math.subtract_6/SubSub(tf.compat.v1.squeeze_10/Squeeze:output:0#tf.math.reduce_mean_5/Mean:output:0*
T0*
_output_shapes
:\
tf.math.reduce_mean/RankRanktf.math.square/Square:y:0*
T0*
_output_shapes
: a
tf.math.reduce_mean/range/startConst*
_output_shapes
: *
dtype0*
value	B : a
tf.math.reduce_mean/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
tf.math.reduce_mean/rangeRange(tf.math.reduce_mean/range/start:output:0!tf.math.reduce_mean/Rank:output:0(tf.math.reduce_mean/range/delta:output:0*#
_output_shapes
:����������
tf.math.reduce_mean/MeanMeantf.math.square/Square:y:0"tf.math.reduce_mean/range:output:0*
T0*
_output_shapes
: `
tf.math.reduce_mean_1/RankRanktf.math.square_1/Square:y:0*
T0*
_output_shapes
: c
!tf.math.reduce_mean_1/range/startConst*
_output_shapes
: *
dtype0*
value	B : c
!tf.math.reduce_mean_1/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
tf.math.reduce_mean_1/rangeRange*tf.math.reduce_mean_1/range/start:output:0#tf.math.reduce_mean_1/Rank:output:0*tf.math.reduce_mean_1/range/delta:output:0*#
_output_shapes
:����������
tf.math.reduce_mean_1/MeanMeantf.math.square_1/Square:y:0$tf.math.reduce_mean_1/range:output:0*
T0*
_output_shapes
: ^
tf.math.maximum/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *��:�
tf.math.maximum/MaximumMaximumtf.math.square_2/Square:y:0"tf.math.maximum/Maximum/y:output:0*
T0*
_output_shapes
:`
tf.math.reduce_mean_3/RankRanktf.math.square_3/Square:y:0*
T0*
_output_shapes
: c
!tf.math.reduce_mean_3/range/startConst*
_output_shapes
: *
dtype0*
value	B : c
!tf.math.reduce_mean_3/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
tf.math.reduce_mean_3/rangeRange*tf.math.reduce_mean_3/range/start:output:0#tf.math.reduce_mean_3/Rank:output:0*tf.math.reduce_mean_3/range/delta:output:0*#
_output_shapes
:����������
tf.math.reduce_mean_3/MeanMeantf.math.square_3/Square:y:0$tf.math.reduce_mean_3/range:output:0*
T0*
_output_shapes
: `
tf.math.reduce_mean_4/RankRanktf.math.square_4/Square:y:0*
T0*
_output_shapes
: c
!tf.math.reduce_mean_4/range/startConst*
_output_shapes
: *
dtype0*
value	B : c
!tf.math.reduce_mean_4/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
tf.math.reduce_mean_4/rangeRange*tf.math.reduce_mean_4/range/start:output:0#tf.math.reduce_mean_4/Rank:output:0*tf.math.reduce_mean_4/range/delta:output:0*#
_output_shapes
:����������
tf.math.reduce_mean_4/MeanMeantf.math.square_4/Square:y:0$tf.math.reduce_mean_4/range:output:0*
T0*
_output_shapes
: y
tf.math.multiply_3/MulMultf.math.subtract_9/Sub:z:0tf.math.subtract_10/Sub:z:0*
T0*
_output_shapes
:x
tf.math.multiply_1/MulMultf.math.subtract_6/Sub:z:0tf.math.subtract_7/Sub:z:0*
T0*
_output_shapes
:a
tf.math.square_8/SquareSquaretf.math.subtract_10/Sub:z:0*
T0*
_output_shapes
:`
tf.math.square_7/SquareSquaretf.math.subtract_9/Sub:z:0*
T0*
_output_shapes
:`
tf.math.square_6/SquareSquaretf.math.subtract_7/Sub:z:0*
T0*
_output_shapes
:`
tf.math.square_5/SquareSquaretf.math.subtract_6/Sub:z:0*
T0*
_output_shapes
:�
tf.__operators__.add/AddV2AddV2!tf.math.reduce_mean/Mean:output:0#tf.math.reduce_mean_1/Mean:output:0*
T0*
_output_shapes
: ]
tf.math.subtract_3/Sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *��:�
tf.math.subtract_3/SubSubtf.math.maximum/Maximum:z:0!tf.math.subtract_3/Sub/y:output:0*
T0*
_output_shapes
:�
tf.__operators__.add_2/AddV2AddV2#tf.math.reduce_mean_3/Mean:output:0#tf.math.reduce_mean_4/Mean:output:0*
T0*
_output_shapes
: ^
tf.math.reduce_sum_3/RankRanktf.math.multiply_3/Mul:z:0*
T0*
_output_shapes
: b
 tf.math.reduce_sum_3/range/startConst*
_output_shapes
: *
dtype0*
value	B : b
 tf.math.reduce_sum_3/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
tf.math.reduce_sum_3/rangeRange)tf.math.reduce_sum_3/range/start:output:0"tf.math.reduce_sum_3/Rank:output:0)tf.math.reduce_sum_3/range/delta:output:0*#
_output_shapes
:����������
tf.math.reduce_sum_3/SumSumtf.math.multiply_3/Mul:z:0#tf.math.reduce_sum_3/range:output:0*
T0*
_output_shapes
: \
tf.math.reduce_sum/RankRanktf.math.multiply_1/Mul:z:0*
T0*
_output_shapes
: `
tf.math.reduce_sum/range/startConst*
_output_shapes
: *
dtype0*
value	B : `
tf.math.reduce_sum/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
tf.math.reduce_sum/rangeRange'tf.math.reduce_sum/range/start:output:0 tf.math.reduce_sum/Rank:output:0'tf.math.reduce_sum/range/delta:output:0*#
_output_shapes
:���������}
tf.math.reduce_sum/SumSumtf.math.multiply_1/Mul:z:0!tf.math.reduce_sum/range:output:0*
T0*
_output_shapes
: _
tf.math.reduce_sum_4/RankRanktf.math.square_7/Square:y:0*
T0*
_output_shapes
: b
 tf.math.reduce_sum_4/range/startConst*
_output_shapes
: *
dtype0*
value	B : b
 tf.math.reduce_sum_4/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
tf.math.reduce_sum_4/rangeRange)tf.math.reduce_sum_4/range/start:output:0"tf.math.reduce_sum_4/Rank:output:0)tf.math.reduce_sum_4/range/delta:output:0*#
_output_shapes
:����������
tf.math.reduce_sum_4/SumSumtf.math.square_7/Square:y:0#tf.math.reduce_sum_4/range:output:0*
T0*
_output_shapes
: _
tf.math.reduce_sum_5/RankRanktf.math.square_8/Square:y:0*
T0*
_output_shapes
: b
 tf.math.reduce_sum_5/range/startConst*
_output_shapes
: *
dtype0*
value	B : b
 tf.math.reduce_sum_5/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
tf.math.reduce_sum_5/rangeRange)tf.math.reduce_sum_5/range/start:output:0"tf.math.reduce_sum_5/Rank:output:0)tf.math.reduce_sum_5/range/delta:output:0*#
_output_shapes
:����������
tf.math.reduce_sum_5/SumSumtf.math.square_8/Square:y:0#tf.math.reduce_sum_5/range:output:0*
T0*
_output_shapes
: _
tf.math.reduce_sum_1/RankRanktf.math.square_5/Square:y:0*
T0*
_output_shapes
: b
 tf.math.reduce_sum_1/range/startConst*
_output_shapes
: *
dtype0*
value	B : b
 tf.math.reduce_sum_1/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
tf.math.reduce_sum_1/rangeRange)tf.math.reduce_sum_1/range/start:output:0"tf.math.reduce_sum_1/Rank:output:0)tf.math.reduce_sum_1/range/delta:output:0*#
_output_shapes
:����������
tf.math.reduce_sum_1/SumSumtf.math.square_5/Square:y:0#tf.math.reduce_sum_1/range:output:0*
T0*
_output_shapes
: _
tf.math.reduce_sum_2/RankRanktf.math.square_6/Square:y:0*
T0*
_output_shapes
: b
 tf.math.reduce_sum_2/range/startConst*
_output_shapes
: *
dtype0*
value	B : b
 tf.math.reduce_sum_2/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
tf.math.reduce_sum_2/rangeRange)tf.math.reduce_sum_2/range/start:output:0"tf.math.reduce_sum_2/Rank:output:0)tf.math.reduce_sum_2/range/delta:output:0*#
_output_shapes
:����������
tf.math.reduce_sum_2/SumSumtf.math.square_6/Square:y:0#tf.math.reduce_sum_2/range:output:0*
T0*
_output_shapes
: _
tf.math.reduce_mean_2/RankRanktf.math.subtract_3/Sub:z:0*
T0*
_output_shapes
: c
!tf.math.reduce_mean_2/range/startConst*
_output_shapes
: *
dtype0*
value	B : c
!tf.math.reduce_mean_2/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
tf.math.reduce_mean_2/rangeRange*tf.math.reduce_mean_2/range/start:output:0#tf.math.reduce_mean_2/Rank:output:0*tf.math.reduce_mean_2/range/delta:output:0*#
_output_shapes
:����������
tf.math.reduce_mean_2/MeanMeantf.math.subtract_3/Sub:z:0$tf.math.reduce_mean_2/range:output:0*
T0*
_output_shapes
: Q
add_metric/RankConst*
_output_shapes
: *
dtype0*
value	B : X
add_metric/range/startConst*
_output_shapes
: *
dtype0*
value	B : X
add_metric/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
add_metric/rangeRangeadd_metric/range/start:output:0add_metric/Rank:output:0add_metric/range/delta:output:0*
_output_shapes
: s
add_metric/SumSum tf.__operators__.add_2/AddV2:z:0add_metric/range:output:0*
T0*
_output_shapes
: �
add_metric/AssignAddVariableOpAssignAddVariableOp'add_metric_assignaddvariableop_resourceadd_metric/Sum:output:0*
_output_shapes
 *
dtype0Q
add_metric/SizeConst*
_output_shapes
: *
dtype0*
value	B :a
add_metric/CastCastadd_metric/Size:output:0*

DstT0*

SrcT0*
_output_shapes
: �
 add_metric/AssignAddVariableOp_1AssignAddVariableOp)add_metric_assignaddvariableop_1_resourceadd_metric/Cast:y:0^add_metric/AssignAddVariableOp*
_output_shapes
 *
dtype0�
$add_metric/div_no_nan/ReadVariableOpReadVariableOp'add_metric_assignaddvariableop_resource^add_metric/AssignAddVariableOp!^add_metric/AssignAddVariableOp_1*
_output_shapes
: *
dtype0�
&add_metric/div_no_nan/ReadVariableOp_1ReadVariableOp)add_metric_assignaddvariableop_1_resource!^add_metric/AssignAddVariableOp_1*
_output_shapes
: *
dtype0�
add_metric/div_no_nanDivNoNan,add_metric/div_no_nan/ReadVariableOp:value:0.add_metric/div_no_nan/ReadVariableOp_1:value:0*
T0*
_output_shapes
: [
add_metric/IdentityIdentityadd_metric/div_no_nan:z:0*
T0*
_output_shapes
: �
tf.math.multiply_4/MulMul!tf.math.reduce_sum_4/Sum:output:0!tf.math.reduce_sum_5/Sum:output:0*
T0*
_output_shapes
: �
tf.math.multiply_2/MulMul!tf.math.reduce_sum_1/Sum:output:0!tf.math.reduce_sum_2/Sum:output:0*
T0*
_output_shapes
: j
tf.math.multiply/MulMulunknown#tf.math.reduce_mean_2/Mean:output:0*
T0*
_output_shapes
: X
tf.math.sqrt_1/SqrtSqrttf.math.multiply_4/Mul:z:0*
T0*
_output_shapes
: V
tf.math.sqrt/SqrtSqrttf.math.multiply_2/Mul:z:0*
T0*
_output_shapes
: �
tf.__operators__.add_1/AddV2AddV2tf.__operators__.add/AddV2:z:0tf.math.multiply/Mul:z:0*
T0*
_output_shapes
: ]
tf.__operators__.add_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *��'7�
tf.__operators__.add_4/AddV2AddV2tf.math.sqrt_1/Sqrt:y:0!tf.__operators__.add_4/y:output:0*
T0*
_output_shapes
: ]
tf.__operators__.add_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *��'7�
tf.__operators__.add_3/AddV2AddV2tf.math.sqrt/Sqrt:y:0!tf.__operators__.add_3/y:output:0*
T0*
_output_shapes
: �
tf.math.truediv_1/truedivRealDiv!tf.math.reduce_sum_3/Sum:output:0 tf.__operators__.add_4/AddV2:z:0*
T0*
_output_shapes
: �
tf.math.truediv/truedivRealDivtf.math.reduce_sum/Sum:output:0 tf.__operators__.add_3/AddV2:z:0*
T0*
_output_shapes
: i
tf.math.subtract_11/SubSub	unknown_0tf.math.truediv_1/truediv:z:0*
T0*
_output_shapes
: f
tf.math.subtract_8/SubSub	unknown_1tf.math.truediv/truediv:z:0*
T0*
_output_shapes
: V
tf.math.pow/Pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?o
tf.math.pow/PowPowtf.math.subtract_8/Sub:z:0tf.math.pow/Pow/y:output:0*
T0*
_output_shapes
: X
tf.math.pow_1/Pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?t
tf.math.pow_1/PowPowtf.math.subtract_11/Sub:z:0tf.math.pow_1/Pow/y:output:0*
T0*
_output_shapes
: r
tf.__operators__.add_5/AddV2AddV2tf.math.pow/Pow:z:0tf.math.pow_1/Pow:z:0*
T0*
_output_shapes
: S
add_metric_1/RankConst*
_output_shapes
: *
dtype0*
value	B : Z
add_metric_1/range/startConst*
_output_shapes
: *
dtype0*
value	B : Z
add_metric_1/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
add_metric_1/rangeRange!add_metric_1/range/start:output:0add_metric_1/Rank:output:0!add_metric_1/range/delta:output:0*
_output_shapes
: w
add_metric_1/SumSum tf.__operators__.add_5/AddV2:z:0add_metric_1/range:output:0*
T0*
_output_shapes
: �
 add_metric_1/AssignAddVariableOpAssignAddVariableOp)add_metric_1_assignaddvariableop_resourceadd_metric_1/Sum:output:0*
_output_shapes
 *
dtype0S
add_metric_1/SizeConst*
_output_shapes
: *
dtype0*
value	B :e
add_metric_1/CastCastadd_metric_1/Size:output:0*

DstT0*

SrcT0*
_output_shapes
: �
"add_metric_1/AssignAddVariableOp_1AssignAddVariableOp+add_metric_1_assignaddvariableop_1_resourceadd_metric_1/Cast:y:0!^add_metric_1/AssignAddVariableOp*
_output_shapes
 *
dtype0�
&add_metric_1/div_no_nan/ReadVariableOpReadVariableOp)add_metric_1_assignaddvariableop_resource!^add_metric_1/AssignAddVariableOp#^add_metric_1/AssignAddVariableOp_1*
_output_shapes
: *
dtype0�
(add_metric_1/div_no_nan/ReadVariableOp_1ReadVariableOp+add_metric_1_assignaddvariableop_1_resource#^add_metric_1/AssignAddVariableOp_1*
_output_shapes
: *
dtype0�
add_metric_1/div_no_nanDivNoNan.add_metric_1/div_no_nan/ReadVariableOp:value:00add_metric_1/div_no_nan/ReadVariableOp_1:value:0*
T0*
_output_shapes
: _
add_metric_1/IdentityIdentityadd_metric_1/div_no_nan:z:0*
T0*
_output_shapes
: �
/conv3d/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+model_conv3d_conv3d_readvariableop_resource**
_output_shapes
:@*
dtype0�
 conv3d/kernel/Regularizer/SquareSquare7conv3d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0**
_output_shapes
:@|
conv3d/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
conv3d/kernel/Regularizer/SumSum$conv3d/kernel/Regularizer/Square:y:0(conv3d/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: d
conv3d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    �
conv3d/kernel/Regularizer/mulMul(conv3d/kernel/Regularizer/mul/x:output:0&conv3d/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1conv3d_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp-model_conv3d_1_conv3d_readvariableop_resource**
_output_shapes
:@@*
dtype0�
"conv3d_1/kernel/Regularizer/SquareSquare9conv3d_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0**
_output_shapes
:@@~
!conv3d_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
conv3d_1/kernel/Regularizer/SumSum&conv3d_1/kernel/Regularizer/Square:y:0*conv3d_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!conv3d_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    �
conv3d_1/kernel/Regularizer/mulMul*conv3d_1/kernel/Regularizer/mul/x:output:0(conv3d_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1conv3d_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp-model_conv3d_2_conv3d_readvariableop_resource*+
_output_shapes
:@�*
dtype0�
"conv3d_2/kernel/Regularizer/SquareSquare9conv3d_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*+
_output_shapes
:@�~
!conv3d_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
conv3d_2/kernel/Regularizer/SumSum&conv3d_2/kernel/Regularizer/Square:y:0*conv3d_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!conv3d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    �
conv3d_2/kernel/Regularizer/mulMul*conv3d_2/kernel/Regularizer/mul/x:output:0(conv3d_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1conv3d_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOp-model_conv3d_3_conv3d_readvariableop_resource*,
_output_shapes
:��*
dtype0�
"conv3d_3/kernel/Regularizer/SquareSquare9conv3d_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*,
_output_shapes
:��~
!conv3d_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
conv3d_3/kernel/Regularizer/SumSum&conv3d_3/kernel/Regularizer/Square:y:0*conv3d_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!conv3d_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    �
conv3d_3/kernel/Regularizer/mulMul*conv3d_3/kernel/Regularizer/mul/x:output:0(conv3d_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1conv3d_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOp-model_conv3d_4_conv3d_readvariableop_resource*,
_output_shapes
:��*
dtype0�
"conv3d_4/kernel/Regularizer/SquareSquare9conv3d_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*,
_output_shapes
:��~
!conv3d_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
conv3d_4/kernel/Regularizer/SumSum&conv3d_4/kernel/Regularizer/Square:y:0*conv3d_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!conv3d_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    �
conv3d_4/kernel/Regularizer/mulMul*conv3d_4/kernel/Regularizer/mul/x:output:0(conv3d_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1conv3d_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOp-model_conv3d_5_conv3d_readvariableop_resource*,
_output_shapes
:��*
dtype0�
"conv3d_5/kernel/Regularizer/SquareSquare9conv3d_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*,
_output_shapes
:��~
!conv3d_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
conv3d_5/kernel/Regularizer/SumSum&conv3d_5/kernel/Regularizer/Square:y:0*conv3d_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!conv3d_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    �
conv3d_5/kernel/Regularizer/mulMul*conv3d_5/kernel/Regularizer/mul/x:output:0(conv3d_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1conv3d_6/kernel/Regularizer/Square/ReadVariableOpReadVariableOp-model_conv3d_6_conv3d_readvariableop_resource*,
_output_shapes
:��*
dtype0�
"conv3d_6/kernel/Regularizer/SquareSquare9conv3d_6/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*,
_output_shapes
:��~
!conv3d_6/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
conv3d_6/kernel/Regularizer/SumSum&conv3d_6/kernel/Regularizer/Square:y:0*conv3d_6/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!conv3d_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    �
conv3d_6/kernel/Regularizer/mulMul*conv3d_6/kernel/Regularizer/mul/x:output:0(conv3d_6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1conv3d_7/kernel/Regularizer/Square/ReadVariableOpReadVariableOp-model_conv3d_7_conv3d_readvariableop_resource*,
_output_shapes
:��*
dtype0�
"conv3d_7/kernel/Regularizer/SquareSquare9conv3d_7/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*,
_output_shapes
:��~
!conv3d_7/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
conv3d_7/kernel/Regularizer/SumSum&conv3d_7/kernel/Regularizer/Square:y:0*conv3d_7/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!conv3d_7/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    �
conv3d_7/kernel/Regularizer/mulMul*conv3d_7/kernel/Regularizer/mul/x:output:0(conv3d_7/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOp*model_dense_matmul_readvariableop_resource*
_output_shapes
:	�$*
dtype0�
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	�$o
dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
dense/kernel/Regularizer/SumSum#dense/kernel/Regularizer/Square:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: c
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    �
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: m
IdentityIdentitymodel/dense/BiasAdd_1:output:0^NoOp*
T0*'
_output_shapes
:���������m

Identity_1Identitymodel/dense/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������`

Identity_2Identity tf.__operators__.add_1/AddV2:z:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp^add_metric/AssignAddVariableOp!^add_metric/AssignAddVariableOp_1%^add_metric/div_no_nan/ReadVariableOp'^add_metric/div_no_nan/ReadVariableOp_1!^add_metric_1/AssignAddVariableOp#^add_metric_1/AssignAddVariableOp_1'^add_metric_1/div_no_nan/ReadVariableOp)^add_metric_1/div_no_nan/ReadVariableOp_10^conv3d/kernel/Regularizer/Square/ReadVariableOp2^conv3d_1/kernel/Regularizer/Square/ReadVariableOp2^conv3d_2/kernel/Regularizer/Square/ReadVariableOp2^conv3d_3/kernel/Regularizer/Square/ReadVariableOp2^conv3d_4/kernel/Regularizer/Square/ReadVariableOp2^conv3d_5/kernel/Regularizer/Square/ReadVariableOp2^conv3d_6/kernel/Regularizer/Square/ReadVariableOp2^conv3d_7/kernel/Regularizer/Square/ReadVariableOp/^dense/kernel/Regularizer/Square/ReadVariableOp$^model/conv3d/BiasAdd/ReadVariableOp&^model/conv3d/BiasAdd_1/ReadVariableOp#^model/conv3d/Conv3D/ReadVariableOp%^model/conv3d/Conv3D_1/ReadVariableOp&^model/conv3d_1/BiasAdd/ReadVariableOp(^model/conv3d_1/BiasAdd_1/ReadVariableOp%^model/conv3d_1/Conv3D/ReadVariableOp'^model/conv3d_1/Conv3D_1/ReadVariableOp&^model/conv3d_2/BiasAdd/ReadVariableOp(^model/conv3d_2/BiasAdd_1/ReadVariableOp%^model/conv3d_2/Conv3D/ReadVariableOp'^model/conv3d_2/Conv3D_1/ReadVariableOp&^model/conv3d_3/BiasAdd/ReadVariableOp(^model/conv3d_3/BiasAdd_1/ReadVariableOp%^model/conv3d_3/Conv3D/ReadVariableOp'^model/conv3d_3/Conv3D_1/ReadVariableOp&^model/conv3d_4/BiasAdd/ReadVariableOp(^model/conv3d_4/BiasAdd_1/ReadVariableOp%^model/conv3d_4/Conv3D/ReadVariableOp'^model/conv3d_4/Conv3D_1/ReadVariableOp&^model/conv3d_5/BiasAdd/ReadVariableOp(^model/conv3d_5/BiasAdd_1/ReadVariableOp%^model/conv3d_5/Conv3D/ReadVariableOp'^model/conv3d_5/Conv3D_1/ReadVariableOp&^model/conv3d_6/BiasAdd/ReadVariableOp(^model/conv3d_6/BiasAdd_1/ReadVariableOp%^model/conv3d_6/Conv3D/ReadVariableOp'^model/conv3d_6/Conv3D_1/ReadVariableOp&^model/conv3d_7/BiasAdd/ReadVariableOp(^model/conv3d_7/BiasAdd_1/ReadVariableOp%^model/conv3d_7/Conv3D/ReadVariableOp'^model/conv3d_7/Conv3D_1/ReadVariableOp#^model/dense/BiasAdd/ReadVariableOp%^model/dense/BiasAdd_1/ReadVariableOp"^model/dense/MatMul/ReadVariableOp$^model/dense/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������22:���������22:���������:���������: : : : : : : : : : : : : : : : : : : : : : : : : 2@
add_metric/AssignAddVariableOpadd_metric/AssignAddVariableOp2D
 add_metric/AssignAddVariableOp_1 add_metric/AssignAddVariableOp_12L
$add_metric/div_no_nan/ReadVariableOp$add_metric/div_no_nan/ReadVariableOp2P
&add_metric/div_no_nan/ReadVariableOp_1&add_metric/div_no_nan/ReadVariableOp_12D
 add_metric_1/AssignAddVariableOp add_metric_1/AssignAddVariableOp2H
"add_metric_1/AssignAddVariableOp_1"add_metric_1/AssignAddVariableOp_12P
&add_metric_1/div_no_nan/ReadVariableOp&add_metric_1/div_no_nan/ReadVariableOp2T
(add_metric_1/div_no_nan/ReadVariableOp_1(add_metric_1/div_no_nan/ReadVariableOp_12b
/conv3d/kernel/Regularizer/Square/ReadVariableOp/conv3d/kernel/Regularizer/Square/ReadVariableOp2f
1conv3d_1/kernel/Regularizer/Square/ReadVariableOp1conv3d_1/kernel/Regularizer/Square/ReadVariableOp2f
1conv3d_2/kernel/Regularizer/Square/ReadVariableOp1conv3d_2/kernel/Regularizer/Square/ReadVariableOp2f
1conv3d_3/kernel/Regularizer/Square/ReadVariableOp1conv3d_3/kernel/Regularizer/Square/ReadVariableOp2f
1conv3d_4/kernel/Regularizer/Square/ReadVariableOp1conv3d_4/kernel/Regularizer/Square/ReadVariableOp2f
1conv3d_5/kernel/Regularizer/Square/ReadVariableOp1conv3d_5/kernel/Regularizer/Square/ReadVariableOp2f
1conv3d_6/kernel/Regularizer/Square/ReadVariableOp1conv3d_6/kernel/Regularizer/Square/ReadVariableOp2f
1conv3d_7/kernel/Regularizer/Square/ReadVariableOp1conv3d_7/kernel/Regularizer/Square/ReadVariableOp2`
.dense/kernel/Regularizer/Square/ReadVariableOp.dense/kernel/Regularizer/Square/ReadVariableOp2J
#model/conv3d/BiasAdd/ReadVariableOp#model/conv3d/BiasAdd/ReadVariableOp2N
%model/conv3d/BiasAdd_1/ReadVariableOp%model/conv3d/BiasAdd_1/ReadVariableOp2H
"model/conv3d/Conv3D/ReadVariableOp"model/conv3d/Conv3D/ReadVariableOp2L
$model/conv3d/Conv3D_1/ReadVariableOp$model/conv3d/Conv3D_1/ReadVariableOp2N
%model/conv3d_1/BiasAdd/ReadVariableOp%model/conv3d_1/BiasAdd/ReadVariableOp2R
'model/conv3d_1/BiasAdd_1/ReadVariableOp'model/conv3d_1/BiasAdd_1/ReadVariableOp2L
$model/conv3d_1/Conv3D/ReadVariableOp$model/conv3d_1/Conv3D/ReadVariableOp2P
&model/conv3d_1/Conv3D_1/ReadVariableOp&model/conv3d_1/Conv3D_1/ReadVariableOp2N
%model/conv3d_2/BiasAdd/ReadVariableOp%model/conv3d_2/BiasAdd/ReadVariableOp2R
'model/conv3d_2/BiasAdd_1/ReadVariableOp'model/conv3d_2/BiasAdd_1/ReadVariableOp2L
$model/conv3d_2/Conv3D/ReadVariableOp$model/conv3d_2/Conv3D/ReadVariableOp2P
&model/conv3d_2/Conv3D_1/ReadVariableOp&model/conv3d_2/Conv3D_1/ReadVariableOp2N
%model/conv3d_3/BiasAdd/ReadVariableOp%model/conv3d_3/BiasAdd/ReadVariableOp2R
'model/conv3d_3/BiasAdd_1/ReadVariableOp'model/conv3d_3/BiasAdd_1/ReadVariableOp2L
$model/conv3d_3/Conv3D/ReadVariableOp$model/conv3d_3/Conv3D/ReadVariableOp2P
&model/conv3d_3/Conv3D_1/ReadVariableOp&model/conv3d_3/Conv3D_1/ReadVariableOp2N
%model/conv3d_4/BiasAdd/ReadVariableOp%model/conv3d_4/BiasAdd/ReadVariableOp2R
'model/conv3d_4/BiasAdd_1/ReadVariableOp'model/conv3d_4/BiasAdd_1/ReadVariableOp2L
$model/conv3d_4/Conv3D/ReadVariableOp$model/conv3d_4/Conv3D/ReadVariableOp2P
&model/conv3d_4/Conv3D_1/ReadVariableOp&model/conv3d_4/Conv3D_1/ReadVariableOp2N
%model/conv3d_5/BiasAdd/ReadVariableOp%model/conv3d_5/BiasAdd/ReadVariableOp2R
'model/conv3d_5/BiasAdd_1/ReadVariableOp'model/conv3d_5/BiasAdd_1/ReadVariableOp2L
$model/conv3d_5/Conv3D/ReadVariableOp$model/conv3d_5/Conv3D/ReadVariableOp2P
&model/conv3d_5/Conv3D_1/ReadVariableOp&model/conv3d_5/Conv3D_1/ReadVariableOp2N
%model/conv3d_6/BiasAdd/ReadVariableOp%model/conv3d_6/BiasAdd/ReadVariableOp2R
'model/conv3d_6/BiasAdd_1/ReadVariableOp'model/conv3d_6/BiasAdd_1/ReadVariableOp2L
$model/conv3d_6/Conv3D/ReadVariableOp$model/conv3d_6/Conv3D/ReadVariableOp2P
&model/conv3d_6/Conv3D_1/ReadVariableOp&model/conv3d_6/Conv3D_1/ReadVariableOp2N
%model/conv3d_7/BiasAdd/ReadVariableOp%model/conv3d_7/BiasAdd/ReadVariableOp2R
'model/conv3d_7/BiasAdd_1/ReadVariableOp'model/conv3d_7/BiasAdd_1/ReadVariableOp2L
$model/conv3d_7/Conv3D/ReadVariableOp$model/conv3d_7/Conv3D/ReadVariableOp2P
&model/conv3d_7/Conv3D_1/ReadVariableOp&model/conv3d_7/Conv3D_1/ReadVariableOp2H
"model/dense/BiasAdd/ReadVariableOp"model/dense/BiasAdd/ReadVariableOp2L
$model/dense/BiasAdd_1/ReadVariableOp$model/dense/BiasAdd_1/ReadVariableOp2F
!model/dense/MatMul/ReadVariableOp!model/dense/MatMul/ReadVariableOp2J
#model/dense/MatMul_1/ReadVariableOp#model/dense/MatMul_1/ReadVariableOp:] Y
3
_output_shapes!
:���������22
"
_user_specified_name
inputs/0:]Y
3
_output_shapes!
:���������22
"
_user_specified_name
inputs/1:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs/2:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs/3:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
�
G__inference_add_metric_1_layer_call_and_return_conditional_losses_14291

inputs&
assignaddvariableop_resource: (
assignaddvariableop_1_resource: 

identity_1��AssignAddVariableOp�AssignAddVariableOp_1�div_no_nan/ReadVariableOp�div_no_nan/ReadVariableOp_1F
RankConst*
_output_shapes
: *
dtype0*
value	B : M
range/startConst*
_output_shapes
: *
dtype0*
value	B : M
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :c
rangeRangerange/start:output:0Rank:output:0range/delta:output:0*
_output_shapes
: C
SumSuminputsrange:output:0*
T0*
_output_shapes
: y
AssignAddVariableOpAssignAddVariableOpassignaddvariableop_resourceSum:output:0*
_output_shapes
 *
dtype0F
SizeConst*
_output_shapes
: *
dtype0*
value	B :K
CastCastSize:output:0*

DstT0*

SrcT0*
_output_shapes
: �
AssignAddVariableOp_1AssignAddVariableOpassignaddvariableop_1_resourceCast:y:0^AssignAddVariableOp*
_output_shapes
 *
dtype0�
div_no_nan/ReadVariableOpReadVariableOpassignaddvariableop_resource^AssignAddVariableOp^AssignAddVariableOp_1*
_output_shapes
: *
dtype0�
div_no_nan/ReadVariableOp_1ReadVariableOpassignaddvariableop_1_resource^AssignAddVariableOp_1*
_output_shapes
: *
dtype0

div_no_nanDivNoNan!div_no_nan/ReadVariableOp:value:0#div_no_nan/ReadVariableOp_1:value:0*
T0*
_output_shapes
: E
IdentityIdentitydiv_no_nan:z:0*
T0*
_output_shapes
: F

Identity_1Identityinputs^NoOp*
T0*
_output_shapes
: �
NoOpNoOp^AssignAddVariableOp^AssignAddVariableOp_1^div_no_nan/ReadVariableOp^div_no_nan/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : 2*
AssignAddVariableOpAssignAddVariableOp2.
AssignAddVariableOp_1AssignAddVariableOp_126
div_no_nan/ReadVariableOpdiv_no_nan/ReadVariableOp2:
div_no_nan/ReadVariableOp_1div_no_nan/ReadVariableOp_1:> :

_output_shapes
: 
 
_user_specified_nameinputs
��
�
B__inference_model_1_layer_call_and_return_conditional_losses_14354

inputs
inputs_1
inputs_2
inputs_3)
model_14047:@
model_14049:@)
model_14051:@@
model_14053:@*
model_14055:@�
model_14057:	�+
model_14059:��
model_14061:	�+
model_14063:��
model_14065:	�+
model_14067:��
model_14069:	�+
model_14071:��
model_14073:	�+
model_14075:��
model_14077:	�
model_14079:	�$
model_14081:
add_metric_14236: 
add_metric_14238: 
unknown
	unknown_0
	unknown_1
add_metric_1_14292: 
add_metric_1_14294: 
identity

identity_1

identity_2��"add_metric/StatefulPartitionedCall�$add_metric_1/StatefulPartitionedCall�/conv3d/kernel/Regularizer/Square/ReadVariableOp�1conv3d_1/kernel/Regularizer/Square/ReadVariableOp�1conv3d_2/kernel/Regularizer/Square/ReadVariableOp�1conv3d_3/kernel/Regularizer/Square/ReadVariableOp�1conv3d_4/kernel/Regularizer/Square/ReadVariableOp�1conv3d_5/kernel/Regularizer/Square/ReadVariableOp�1conv3d_6/kernel/Regularizer/Square/ReadVariableOp�1conv3d_7/kernel/Regularizer/Square/ReadVariableOp�.dense/kernel/Regularizer/Square/ReadVariableOp�model/StatefulPartitionedCall�model/StatefulPartitionedCall_1�
model/StatefulPartitionedCallStatefulPartitionedCallinputs_1model_14047model_14049model_14051model_14053model_14055model_14057model_14059model_14061model_14063model_14065model_14067model_14069model_14071model_14073model_14075model_14077model_14079model_14081*
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
@__inference_model_layer_call_and_return_conditional_losses_13452�
model/StatefulPartitionedCall_1StatefulPartitionedCallinputsmodel_14047model_14049model_14051model_14053model_14055model_14057model_14059model_14061model_14063model_14065model_14067model_14069model_14071model_14073model_14075model_14077model_14079model_14081*
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
@__inference_model_layer_call_and_return_conditional_losses_13452V
tf.compat.v1.squeeze_1/SqueezeSqueezeinputs_3*
T0*
_output_shapes
:t
tf.compat.v1.squeeze_3/SqueezeSqueeze&model/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:T
tf.compat.v1.squeeze/SqueezeSqueezeinputs_2*
T0*
_output_shapes
:v
tf.compat.v1.squeeze_2/SqueezeSqueeze(model/StatefulPartitionedCall_1:output:0*
T0*
_output_shapes
:v
tf.compat.v1.squeeze_4/SqueezeSqueeze(model/StatefulPartitionedCall_1:output:0*
T0*
_output_shapes
:t
tf.compat.v1.squeeze_5/SqueezeSqueeze&model/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:V
tf.compat.v1.squeeze_7/SqueezeSqueezeinputs_3*
T0*
_output_shapes
:t
tf.compat.v1.squeeze_9/SqueezeSqueeze&model/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:V
tf.compat.v1.squeeze_6/SqueezeSqueezeinputs_2*
T0*
_output_shapes
:v
tf.compat.v1.squeeze_8/SqueezeSqueeze(model/StatefulPartitionedCall_1:output:0*
T0*
_output_shapes
:u
tf.compat.v1.squeeze_13/SqueezeSqueeze&model/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:W
tf.compat.v1.squeeze_12/SqueezeSqueezeinputs_3*
T0*
_output_shapes
:w
tf.compat.v1.squeeze_11/SqueezeSqueeze(model/StatefulPartitionedCall_1:output:0*
T0*
_output_shapes
:W
tf.compat.v1.squeeze_10/SqueezeSqueezeinputs_2*
T0*
_output_shapes
:�
tf.math.subtract_1/SubSub'tf.compat.v1.squeeze_1/Squeeze:output:0'tf.compat.v1.squeeze_3/Squeeze:output:0*
T0*
_output_shapes
:�
tf.math.subtract/SubSub%tf.compat.v1.squeeze/Squeeze:output:0'tf.compat.v1.squeeze_2/Squeeze:output:0*
T0*
_output_shapes
:�
tf.math.subtract_2/SubSub'tf.compat.v1.squeeze_4/Squeeze:output:0'tf.compat.v1.squeeze_5/Squeeze:output:0*
T0*
_output_shapes
:�
tf.math.subtract_5/SubSub'tf.compat.v1.squeeze_7/Squeeze:output:0'tf.compat.v1.squeeze_9/Squeeze:output:0*
T0*
_output_shapes
:�
tf.math.subtract_4/SubSub'tf.compat.v1.squeeze_6/Squeeze:output:0'tf.compat.v1.squeeze_8/Squeeze:output:0*
T0*
_output_shapes
:m
tf.math.reduce_mean_8/RankRank(tf.compat.v1.squeeze_13/Squeeze:output:0*
T0*
_output_shapes
: c
!tf.math.reduce_mean_8/range/startConst*
_output_shapes
: *
dtype0*
value	B : c
!tf.math.reduce_mean_8/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
tf.math.reduce_mean_8/rangeRange*tf.math.reduce_mean_8/range/start:output:0#tf.math.reduce_mean_8/Rank:output:0*tf.math.reduce_mean_8/range/delta:output:0*#
_output_shapes
:����������
tf.math.reduce_mean_8/MeanMean(tf.compat.v1.squeeze_13/Squeeze:output:0$tf.math.reduce_mean_8/range:output:0*
T0*
_output_shapes
: m
tf.math.reduce_mean_7/RankRank(tf.compat.v1.squeeze_12/Squeeze:output:0*
T0*
_output_shapes
: c
!tf.math.reduce_mean_7/range/startConst*
_output_shapes
: *
dtype0*
value	B : c
!tf.math.reduce_mean_7/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
tf.math.reduce_mean_7/rangeRange*tf.math.reduce_mean_7/range/start:output:0#tf.math.reduce_mean_7/Rank:output:0*tf.math.reduce_mean_7/range/delta:output:0*#
_output_shapes
:����������
tf.math.reduce_mean_7/MeanMean(tf.compat.v1.squeeze_12/Squeeze:output:0$tf.math.reduce_mean_7/range:output:0*
T0*
_output_shapes
: m
tf.math.reduce_mean_6/RankRank(tf.compat.v1.squeeze_11/Squeeze:output:0*
T0*
_output_shapes
: c
!tf.math.reduce_mean_6/range/startConst*
_output_shapes
: *
dtype0*
value	B : c
!tf.math.reduce_mean_6/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
tf.math.reduce_mean_6/rangeRange*tf.math.reduce_mean_6/range/start:output:0#tf.math.reduce_mean_6/Rank:output:0*tf.math.reduce_mean_6/range/delta:output:0*#
_output_shapes
:����������
tf.math.reduce_mean_6/MeanMean(tf.compat.v1.squeeze_11/Squeeze:output:0$tf.math.reduce_mean_6/range:output:0*
T0*
_output_shapes
: m
tf.math.reduce_mean_5/RankRank(tf.compat.v1.squeeze_10/Squeeze:output:0*
T0*
_output_shapes
: c
!tf.math.reduce_mean_5/range/startConst*
_output_shapes
: *
dtype0*
value	B : c
!tf.math.reduce_mean_5/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
tf.math.reduce_mean_5/rangeRange*tf.math.reduce_mean_5/range/start:output:0#tf.math.reduce_mean_5/Rank:output:0*tf.math.reduce_mean_5/range/delta:output:0*#
_output_shapes
:����������
tf.math.reduce_mean_5/MeanMean(tf.compat.v1.squeeze_10/Squeeze:output:0$tf.math.reduce_mean_5/range:output:0*
T0*
_output_shapes
: `
tf.math.square_1/SquareSquaretf.math.subtract_1/Sub:z:0*
T0*
_output_shapes
:\
tf.math.square/SquareSquaretf.math.subtract/Sub:z:0*
T0*
_output_shapes
:`
tf.math.square_2/SquareSquaretf.math.subtract_2/Sub:z:0*
T0*
_output_shapes
:`
tf.math.square_4/SquareSquaretf.math.subtract_5/Sub:z:0*
T0*
_output_shapes
:`
tf.math.square_3/SquareSquaretf.math.subtract_4/Sub:z:0*
T0*
_output_shapes
:�
tf.math.subtract_10/SubSub(tf.compat.v1.squeeze_13/Squeeze:output:0#tf.math.reduce_mean_8/Mean:output:0*
T0*
_output_shapes
:�
tf.math.subtract_9/SubSub(tf.compat.v1.squeeze_12/Squeeze:output:0#tf.math.reduce_mean_7/Mean:output:0*
T0*
_output_shapes
:�
tf.math.subtract_7/SubSub(tf.compat.v1.squeeze_11/Squeeze:output:0#tf.math.reduce_mean_6/Mean:output:0*
T0*
_output_shapes
:�
tf.math.subtract_6/SubSub(tf.compat.v1.squeeze_10/Squeeze:output:0#tf.math.reduce_mean_5/Mean:output:0*
T0*
_output_shapes
:\
tf.math.reduce_mean/RankRanktf.math.square/Square:y:0*
T0*
_output_shapes
: a
tf.math.reduce_mean/range/startConst*
_output_shapes
: *
dtype0*
value	B : a
tf.math.reduce_mean/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
tf.math.reduce_mean/rangeRange(tf.math.reduce_mean/range/start:output:0!tf.math.reduce_mean/Rank:output:0(tf.math.reduce_mean/range/delta:output:0*#
_output_shapes
:����������
tf.math.reduce_mean/MeanMeantf.math.square/Square:y:0"tf.math.reduce_mean/range:output:0*
T0*
_output_shapes
: `
tf.math.reduce_mean_1/RankRanktf.math.square_1/Square:y:0*
T0*
_output_shapes
: c
!tf.math.reduce_mean_1/range/startConst*
_output_shapes
: *
dtype0*
value	B : c
!tf.math.reduce_mean_1/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
tf.math.reduce_mean_1/rangeRange*tf.math.reduce_mean_1/range/start:output:0#tf.math.reduce_mean_1/Rank:output:0*tf.math.reduce_mean_1/range/delta:output:0*#
_output_shapes
:����������
tf.math.reduce_mean_1/MeanMeantf.math.square_1/Square:y:0$tf.math.reduce_mean_1/range:output:0*
T0*
_output_shapes
: ^
tf.math.maximum/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *��:�
tf.math.maximum/MaximumMaximumtf.math.square_2/Square:y:0"tf.math.maximum/Maximum/y:output:0*
T0*
_output_shapes
:`
tf.math.reduce_mean_3/RankRanktf.math.square_3/Square:y:0*
T0*
_output_shapes
: c
!tf.math.reduce_mean_3/range/startConst*
_output_shapes
: *
dtype0*
value	B : c
!tf.math.reduce_mean_3/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
tf.math.reduce_mean_3/rangeRange*tf.math.reduce_mean_3/range/start:output:0#tf.math.reduce_mean_3/Rank:output:0*tf.math.reduce_mean_3/range/delta:output:0*#
_output_shapes
:����������
tf.math.reduce_mean_3/MeanMeantf.math.square_3/Square:y:0$tf.math.reduce_mean_3/range:output:0*
T0*
_output_shapes
: `
tf.math.reduce_mean_4/RankRanktf.math.square_4/Square:y:0*
T0*
_output_shapes
: c
!tf.math.reduce_mean_4/range/startConst*
_output_shapes
: *
dtype0*
value	B : c
!tf.math.reduce_mean_4/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
tf.math.reduce_mean_4/rangeRange*tf.math.reduce_mean_4/range/start:output:0#tf.math.reduce_mean_4/Rank:output:0*tf.math.reduce_mean_4/range/delta:output:0*#
_output_shapes
:����������
tf.math.reduce_mean_4/MeanMeantf.math.square_4/Square:y:0$tf.math.reduce_mean_4/range:output:0*
T0*
_output_shapes
: y
tf.math.multiply_3/MulMultf.math.subtract_9/Sub:z:0tf.math.subtract_10/Sub:z:0*
T0*
_output_shapes
:x
tf.math.multiply_1/MulMultf.math.subtract_6/Sub:z:0tf.math.subtract_7/Sub:z:0*
T0*
_output_shapes
:a
tf.math.square_8/SquareSquaretf.math.subtract_10/Sub:z:0*
T0*
_output_shapes
:`
tf.math.square_7/SquareSquaretf.math.subtract_9/Sub:z:0*
T0*
_output_shapes
:`
tf.math.square_6/SquareSquaretf.math.subtract_7/Sub:z:0*
T0*
_output_shapes
:`
tf.math.square_5/SquareSquaretf.math.subtract_6/Sub:z:0*
T0*
_output_shapes
:�
tf.__operators__.add/AddV2AddV2!tf.math.reduce_mean/Mean:output:0#tf.math.reduce_mean_1/Mean:output:0*
T0*
_output_shapes
: ]
tf.math.subtract_3/Sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *��:�
tf.math.subtract_3/SubSubtf.math.maximum/Maximum:z:0!tf.math.subtract_3/Sub/y:output:0*
T0*
_output_shapes
:�
tf.__operators__.add_2/AddV2AddV2#tf.math.reduce_mean_3/Mean:output:0#tf.math.reduce_mean_4/Mean:output:0*
T0*
_output_shapes
: ^
tf.math.reduce_sum_3/RankRanktf.math.multiply_3/Mul:z:0*
T0*
_output_shapes
: b
 tf.math.reduce_sum_3/range/startConst*
_output_shapes
: *
dtype0*
value	B : b
 tf.math.reduce_sum_3/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
tf.math.reduce_sum_3/rangeRange)tf.math.reduce_sum_3/range/start:output:0"tf.math.reduce_sum_3/Rank:output:0)tf.math.reduce_sum_3/range/delta:output:0*#
_output_shapes
:����������
tf.math.reduce_sum_3/SumSumtf.math.multiply_3/Mul:z:0#tf.math.reduce_sum_3/range:output:0*
T0*
_output_shapes
: \
tf.math.reduce_sum/RankRanktf.math.multiply_1/Mul:z:0*
T0*
_output_shapes
: `
tf.math.reduce_sum/range/startConst*
_output_shapes
: *
dtype0*
value	B : `
tf.math.reduce_sum/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
tf.math.reduce_sum/rangeRange'tf.math.reduce_sum/range/start:output:0 tf.math.reduce_sum/Rank:output:0'tf.math.reduce_sum/range/delta:output:0*#
_output_shapes
:���������}
tf.math.reduce_sum/SumSumtf.math.multiply_1/Mul:z:0!tf.math.reduce_sum/range:output:0*
T0*
_output_shapes
: _
tf.math.reduce_sum_4/RankRanktf.math.square_7/Square:y:0*
T0*
_output_shapes
: b
 tf.math.reduce_sum_4/range/startConst*
_output_shapes
: *
dtype0*
value	B : b
 tf.math.reduce_sum_4/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
tf.math.reduce_sum_4/rangeRange)tf.math.reduce_sum_4/range/start:output:0"tf.math.reduce_sum_4/Rank:output:0)tf.math.reduce_sum_4/range/delta:output:0*#
_output_shapes
:����������
tf.math.reduce_sum_4/SumSumtf.math.square_7/Square:y:0#tf.math.reduce_sum_4/range:output:0*
T0*
_output_shapes
: _
tf.math.reduce_sum_5/RankRanktf.math.square_8/Square:y:0*
T0*
_output_shapes
: b
 tf.math.reduce_sum_5/range/startConst*
_output_shapes
: *
dtype0*
value	B : b
 tf.math.reduce_sum_5/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
tf.math.reduce_sum_5/rangeRange)tf.math.reduce_sum_5/range/start:output:0"tf.math.reduce_sum_5/Rank:output:0)tf.math.reduce_sum_5/range/delta:output:0*#
_output_shapes
:����������
tf.math.reduce_sum_5/SumSumtf.math.square_8/Square:y:0#tf.math.reduce_sum_5/range:output:0*
T0*
_output_shapes
: _
tf.math.reduce_sum_1/RankRanktf.math.square_5/Square:y:0*
T0*
_output_shapes
: b
 tf.math.reduce_sum_1/range/startConst*
_output_shapes
: *
dtype0*
value	B : b
 tf.math.reduce_sum_1/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
tf.math.reduce_sum_1/rangeRange)tf.math.reduce_sum_1/range/start:output:0"tf.math.reduce_sum_1/Rank:output:0)tf.math.reduce_sum_1/range/delta:output:0*#
_output_shapes
:����������
tf.math.reduce_sum_1/SumSumtf.math.square_5/Square:y:0#tf.math.reduce_sum_1/range:output:0*
T0*
_output_shapes
: _
tf.math.reduce_sum_2/RankRanktf.math.square_6/Square:y:0*
T0*
_output_shapes
: b
 tf.math.reduce_sum_2/range/startConst*
_output_shapes
: *
dtype0*
value	B : b
 tf.math.reduce_sum_2/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
tf.math.reduce_sum_2/rangeRange)tf.math.reduce_sum_2/range/start:output:0"tf.math.reduce_sum_2/Rank:output:0)tf.math.reduce_sum_2/range/delta:output:0*#
_output_shapes
:����������
tf.math.reduce_sum_2/SumSumtf.math.square_6/Square:y:0#tf.math.reduce_sum_2/range:output:0*
T0*
_output_shapes
: _
tf.math.reduce_mean_2/RankRanktf.math.subtract_3/Sub:z:0*
T0*
_output_shapes
: c
!tf.math.reduce_mean_2/range/startConst*
_output_shapes
: *
dtype0*
value	B : c
!tf.math.reduce_mean_2/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
tf.math.reduce_mean_2/rangeRange*tf.math.reduce_mean_2/range/start:output:0#tf.math.reduce_mean_2/Rank:output:0*tf.math.reduce_mean_2/range/delta:output:0*#
_output_shapes
:����������
tf.math.reduce_mean_2/MeanMeantf.math.subtract_3/Sub:z:0$tf.math.reduce_mean_2/range:output:0*
T0*
_output_shapes
: �
"add_metric/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_2/AddV2:z:0add_metric_14236add_metric_14238*
Tin
2*
Tout
2*
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
GPU(2*0J 8� *N
fIRG
E__inference_add_metric_layer_call_and_return_conditional_losses_14235�
tf.math.multiply_4/MulMul!tf.math.reduce_sum_4/Sum:output:0!tf.math.reduce_sum_5/Sum:output:0*
T0*
_output_shapes
: �
tf.math.multiply_2/MulMul!tf.math.reduce_sum_1/Sum:output:0!tf.math.reduce_sum_2/Sum:output:0*
T0*
_output_shapes
: j
tf.math.multiply/MulMulunknown#tf.math.reduce_mean_2/Mean:output:0*
T0*
_output_shapes
: X
tf.math.sqrt_1/SqrtSqrttf.math.multiply_4/Mul:z:0*
T0*
_output_shapes
: V
tf.math.sqrt/SqrtSqrttf.math.multiply_2/Mul:z:0*
T0*
_output_shapes
: �
tf.__operators__.add_1/AddV2AddV2tf.__operators__.add/AddV2:z:0tf.math.multiply/Mul:z:0*
T0*
_output_shapes
: ]
tf.__operators__.add_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *��'7�
tf.__operators__.add_4/AddV2AddV2tf.math.sqrt_1/Sqrt:y:0!tf.__operators__.add_4/y:output:0*
T0*
_output_shapes
: ]
tf.__operators__.add_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *��'7�
tf.__operators__.add_3/AddV2AddV2tf.math.sqrt/Sqrt:y:0!tf.__operators__.add_3/y:output:0*
T0*
_output_shapes
: �
add_loss/PartitionedCallPartitionedCall tf.__operators__.add_1/AddV2:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: : * 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8� *L
fGRE
C__inference_add_loss_layer_call_and_return_conditional_losses_14258�
tf.math.truediv_1/truedivRealDiv!tf.math.reduce_sum_3/Sum:output:0 tf.__operators__.add_4/AddV2:z:0*
T0*
_output_shapes
: �
tf.math.truediv/truedivRealDivtf.math.reduce_sum/Sum:output:0 tf.__operators__.add_3/AddV2:z:0*
T0*
_output_shapes
: i
tf.math.subtract_11/SubSub	unknown_0tf.math.truediv_1/truediv:z:0*
T0*
_output_shapes
: f
tf.math.subtract_8/SubSub	unknown_1tf.math.truediv/truediv:z:0*
T0*
_output_shapes
: V
tf.math.pow/Pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?o
tf.math.pow/PowPowtf.math.subtract_8/Sub:z:0tf.math.pow/Pow/y:output:0*
T0*
_output_shapes
: X
tf.math.pow_1/Pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?t
tf.math.pow_1/PowPowtf.math.subtract_11/Sub:z:0tf.math.pow_1/Pow/y:output:0*
T0*
_output_shapes
: r
tf.__operators__.add_5/AddV2AddV2tf.math.pow/Pow:z:0tf.math.pow_1/Pow:z:0*
T0*
_output_shapes
: �
$add_metric_1/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_5/AddV2:z:0add_metric_1_14292add_metric_1_14294*
Tin
2*
Tout
2*
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
GPU(2*0J 8� *P
fKRI
G__inference_add_metric_1_layer_call_and_return_conditional_losses_14291�
/conv3d/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmodel_14047**
_output_shapes
:@*
dtype0�
 conv3d/kernel/Regularizer/SquareSquare7conv3d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0**
_output_shapes
:@|
conv3d/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
conv3d/kernel/Regularizer/SumSum$conv3d/kernel/Regularizer/Square:y:0(conv3d/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: d
conv3d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    �
conv3d/kernel/Regularizer/mulMul(conv3d/kernel/Regularizer/mul/x:output:0&conv3d/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1conv3d_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmodel_14051**
_output_shapes
:@@*
dtype0�
"conv3d_1/kernel/Regularizer/SquareSquare9conv3d_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0**
_output_shapes
:@@~
!conv3d_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
conv3d_1/kernel/Regularizer/SumSum&conv3d_1/kernel/Regularizer/Square:y:0*conv3d_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!conv3d_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    �
conv3d_1/kernel/Regularizer/mulMul*conv3d_1/kernel/Regularizer/mul/x:output:0(conv3d_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1conv3d_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmodel_14055*+
_output_shapes
:@�*
dtype0�
"conv3d_2/kernel/Regularizer/SquareSquare9conv3d_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*+
_output_shapes
:@�~
!conv3d_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
conv3d_2/kernel/Regularizer/SumSum&conv3d_2/kernel/Regularizer/Square:y:0*conv3d_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!conv3d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    �
conv3d_2/kernel/Regularizer/mulMul*conv3d_2/kernel/Regularizer/mul/x:output:0(conv3d_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1conv3d_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmodel_14059*,
_output_shapes
:��*
dtype0�
"conv3d_3/kernel/Regularizer/SquareSquare9conv3d_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*,
_output_shapes
:��~
!conv3d_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
conv3d_3/kernel/Regularizer/SumSum&conv3d_3/kernel/Regularizer/Square:y:0*conv3d_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!conv3d_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    �
conv3d_3/kernel/Regularizer/mulMul*conv3d_3/kernel/Regularizer/mul/x:output:0(conv3d_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1conv3d_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmodel_14063*,
_output_shapes
:��*
dtype0�
"conv3d_4/kernel/Regularizer/SquareSquare9conv3d_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*,
_output_shapes
:��~
!conv3d_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
conv3d_4/kernel/Regularizer/SumSum&conv3d_4/kernel/Regularizer/Square:y:0*conv3d_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!conv3d_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    �
conv3d_4/kernel/Regularizer/mulMul*conv3d_4/kernel/Regularizer/mul/x:output:0(conv3d_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1conv3d_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmodel_14067*,
_output_shapes
:��*
dtype0�
"conv3d_5/kernel/Regularizer/SquareSquare9conv3d_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*,
_output_shapes
:��~
!conv3d_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
conv3d_5/kernel/Regularizer/SumSum&conv3d_5/kernel/Regularizer/Square:y:0*conv3d_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!conv3d_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    �
conv3d_5/kernel/Regularizer/mulMul*conv3d_5/kernel/Regularizer/mul/x:output:0(conv3d_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1conv3d_6/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmodel_14071*,
_output_shapes
:��*
dtype0�
"conv3d_6/kernel/Regularizer/SquareSquare9conv3d_6/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*,
_output_shapes
:��~
!conv3d_6/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
conv3d_6/kernel/Regularizer/SumSum&conv3d_6/kernel/Regularizer/Square:y:0*conv3d_6/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!conv3d_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    �
conv3d_6/kernel/Regularizer/mulMul*conv3d_6/kernel/Regularizer/mul/x:output:0(conv3d_6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1conv3d_7/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmodel_14075*,
_output_shapes
:��*
dtype0�
"conv3d_7/kernel/Regularizer/SquareSquare9conv3d_7/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*,
_output_shapes
:��~
!conv3d_7/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
conv3d_7/kernel/Regularizer/SumSum&conv3d_7/kernel/Regularizer/Square:y:0*conv3d_7/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!conv3d_7/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    �
conv3d_7/kernel/Regularizer/mulMul*conv3d_7/kernel/Regularizer/mul/x:output:0(conv3d_7/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: {
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmodel_14079*
_output_shapes
:	�$*
dtype0�
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	�$o
dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
dense/kernel/Regularizer/SumSum#dense/kernel/Regularizer/Square:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: c
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    �
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: w
IdentityIdentity(model/StatefulPartitionedCall_1:output:0^NoOp*
T0*'
_output_shapes
:���������w

Identity_1Identity&model/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������a

Identity_2Identity!add_loss/PartitionedCall:output:1^NoOp*
T0*
_output_shapes
: �
NoOpNoOp#^add_metric/StatefulPartitionedCall%^add_metric_1/StatefulPartitionedCall0^conv3d/kernel/Regularizer/Square/ReadVariableOp2^conv3d_1/kernel/Regularizer/Square/ReadVariableOp2^conv3d_2/kernel/Regularizer/Square/ReadVariableOp2^conv3d_3/kernel/Regularizer/Square/ReadVariableOp2^conv3d_4/kernel/Regularizer/Square/ReadVariableOp2^conv3d_5/kernel/Regularizer/Square/ReadVariableOp2^conv3d_6/kernel/Regularizer/Square/ReadVariableOp2^conv3d_7/kernel/Regularizer/Square/ReadVariableOp/^dense/kernel/Regularizer/Square/ReadVariableOp^model/StatefulPartitionedCall ^model/StatefulPartitionedCall_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������22:���������22:���������:���������: : : : : : : : : : : : : : : : : : : : : : : : : 2H
"add_metric/StatefulPartitionedCall"add_metric/StatefulPartitionedCall2L
$add_metric_1/StatefulPartitionedCall$add_metric_1/StatefulPartitionedCall2b
/conv3d/kernel/Regularizer/Square/ReadVariableOp/conv3d/kernel/Regularizer/Square/ReadVariableOp2f
1conv3d_1/kernel/Regularizer/Square/ReadVariableOp1conv3d_1/kernel/Regularizer/Square/ReadVariableOp2f
1conv3d_2/kernel/Regularizer/Square/ReadVariableOp1conv3d_2/kernel/Regularizer/Square/ReadVariableOp2f
1conv3d_3/kernel/Regularizer/Square/ReadVariableOp1conv3d_3/kernel/Regularizer/Square/ReadVariableOp2f
1conv3d_4/kernel/Regularizer/Square/ReadVariableOp1conv3d_4/kernel/Regularizer/Square/ReadVariableOp2f
1conv3d_5/kernel/Regularizer/Square/ReadVariableOp1conv3d_5/kernel/Regularizer/Square/ReadVariableOp2f
1conv3d_6/kernel/Regularizer/Square/ReadVariableOp1conv3d_6/kernel/Regularizer/Square/ReadVariableOp2f
1conv3d_7/kernel/Regularizer/Square/ReadVariableOp1conv3d_7/kernel/Regularizer/Square/ReadVariableOp2`
.dense/kernel/Regularizer/Square/ReadVariableOp.dense/kernel/Regularizer/Square/ReadVariableOp2>
model/StatefulPartitionedCallmodel/StatefulPartitionedCall2B
model/StatefulPartitionedCall_1model/StatefulPartitionedCall_1:[ W
3
_output_shapes!
:���������22
 
_user_specified_nameinputs:[W
3
_output_shapes!
:���������22
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
�
'__inference_model_1_layer_call_fn_15670
inputs_0
inputs_1
inputs_2
inputs_3%
unknown:@
	unknown_0:@'
	unknown_1:@@
	unknown_2:@(
	unknown_3:@�
	unknown_4:	�)
	unknown_5:��
	unknown_6:	�)
	unknown_7:��
	unknown_8:	�)
	unknown_9:��

unknown_10:	�*

unknown_11:��

unknown_12:	�*

unknown_13:��

unknown_14:	�

unknown_15:	�$

unknown_16:

unknown_17: 

unknown_18: 

unknown_19

unknown_20

unknown_21

unknown_22: 

unknown_23: 
identity

identity_1��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1inputs_2inputs_3unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23*(
Tin!
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:���������:���������: *4
_read_only_resource_inputs
	
*4
config_proto$"

CPU

GPU(2*0J 8� *K
fFRD
B__inference_model_1_layer_call_and_return_conditional_losses_14354o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������22:���������22:���������:���������: : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
3
_output_shapes!
:���������22
"
_user_specified_name
inputs/0:]Y
3
_output_shapes!
:���������22
"
_user_specified_name
inputs/1:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs/2:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs/3:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
�
E__inference_add_metric_layer_call_and_return_conditional_losses_14235

inputs&
assignaddvariableop_resource: (
assignaddvariableop_1_resource: 

identity_1��AssignAddVariableOp�AssignAddVariableOp_1�div_no_nan/ReadVariableOp�div_no_nan/ReadVariableOp_1F
RankConst*
_output_shapes
: *
dtype0*
value	B : M
range/startConst*
_output_shapes
: *
dtype0*
value	B : M
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :c
rangeRangerange/start:output:0Rank:output:0range/delta:output:0*
_output_shapes
: C
SumSuminputsrange:output:0*
T0*
_output_shapes
: y
AssignAddVariableOpAssignAddVariableOpassignaddvariableop_resourceSum:output:0*
_output_shapes
 *
dtype0F
SizeConst*
_output_shapes
: *
dtype0*
value	B :K
CastCastSize:output:0*

DstT0*

SrcT0*
_output_shapes
: �
AssignAddVariableOp_1AssignAddVariableOpassignaddvariableop_1_resourceCast:y:0^AssignAddVariableOp*
_output_shapes
 *
dtype0�
div_no_nan/ReadVariableOpReadVariableOpassignaddvariableop_resource^AssignAddVariableOp^AssignAddVariableOp_1*
_output_shapes
: *
dtype0�
div_no_nan/ReadVariableOp_1ReadVariableOpassignaddvariableop_1_resource^AssignAddVariableOp_1*
_output_shapes
: *
dtype0

div_no_nanDivNoNan!div_no_nan/ReadVariableOp:value:0#div_no_nan/ReadVariableOp_1:value:0*
T0*
_output_shapes
: E
IdentityIdentitydiv_no_nan:z:0*
T0*
_output_shapes
: F

Identity_1Identityinputs^NoOp*
T0*
_output_shapes
: �
NoOpNoOp^AssignAddVariableOp^AssignAddVariableOp_1^div_no_nan/ReadVariableOp^div_no_nan/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : 2*
AssignAddVariableOpAssignAddVariableOp2.
AssignAddVariableOp_1AssignAddVariableOp_126
div_no_nan/ReadVariableOpdiv_no_nan/ReadVariableOp2:
div_no_nan/ReadVariableOp_1div_no_nan/ReadVariableOp_1:> :

_output_shapes
: 
 
_user_specified_nameinputs
�
�
C__inference_conv3d_5_layer_call_and_return_conditional_losses_17052

inputs>
conv3d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv3D/ReadVariableOp�1conv3d_5/kernel/Regularizer/Square/ReadVariableOp�
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource*,
_output_shapes
:��*
dtype0�
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :����������*
paddingSAME*
strides	
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :����������]
ReluReluBiasAdd:output:0*
T0*4
_output_shapes"
 :�����������
1conv3d_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv3d_readvariableop_resource*,
_output_shapes
:��*
dtype0�
"conv3d_5/kernel/Regularizer/SquareSquare9conv3d_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*,
_output_shapes
:��~
!conv3d_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
conv3d_5/kernel/Regularizer/SumSum&conv3d_5/kernel/Regularizer/Square:y:0*conv3d_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!conv3d_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    �
conv3d_5/kernel/Regularizer/mulMul*conv3d_5/kernel/Regularizer/mul/x:output:0(conv3d_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: n
IdentityIdentityRelu:activations:0^NoOp*
T0*4
_output_shapes"
 :�����������
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp2^conv3d_5/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp2f
1conv3d_5/kernel/Regularizer/Square/ReadVariableOp1conv3d_5/kernel/Regularizer/Square/ReadVariableOp:\ X
4
_output_shapes"
 :����������
 
_user_specified_nameinputs
�
d
H__inference_max_pooling3d_layer_call_and_return_conditional_losses_13133

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
�
�
(__inference_conv3d_3_layer_call_fn_16973

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
 :���������
�*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *L
fGRE
C__inference_conv3d_3_layer_call_and_return_conditional_losses_13266|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :���������
�`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:���������
�: : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :���������
�
 
_user_specified_nameinputs
ٚ
�
 __inference__wrapped_model_13124
input_2
input_3
input_4
input_5Q
3model_1_model_conv3d_conv3d_readvariableop_resource:@B
4model_1_model_conv3d_biasadd_readvariableop_resource:@S
5model_1_model_conv3d_1_conv3d_readvariableop_resource:@@D
6model_1_model_conv3d_1_biasadd_readvariableop_resource:@T
5model_1_model_conv3d_2_conv3d_readvariableop_resource:@�E
6model_1_model_conv3d_2_biasadd_readvariableop_resource:	�U
5model_1_model_conv3d_3_conv3d_readvariableop_resource:��E
6model_1_model_conv3d_3_biasadd_readvariableop_resource:	�U
5model_1_model_conv3d_4_conv3d_readvariableop_resource:��E
6model_1_model_conv3d_4_biasadd_readvariableop_resource:	�U
5model_1_model_conv3d_5_conv3d_readvariableop_resource:��E
6model_1_model_conv3d_5_biasadd_readvariableop_resource:	�U
5model_1_model_conv3d_6_conv3d_readvariableop_resource:��E
6model_1_model_conv3d_6_biasadd_readvariableop_resource:	�U
5model_1_model_conv3d_7_conv3d_readvariableop_resource:��E
6model_1_model_conv3d_7_biasadd_readvariableop_resource:	�E
2model_1_model_dense_matmul_readvariableop_resource:	�$A
3model_1_model_dense_biasadd_readvariableop_resource:9
/model_1_add_metric_assignaddvariableop_resource: ;
1model_1_add_metric_assignaddvariableop_1_resource: 
model_1_13086
model_1_13098
model_1_13101;
1model_1_add_metric_1_assignaddvariableop_resource: =
3model_1_add_metric_1_assignaddvariableop_1_resource: 
identity

identity_1��&model_1/add_metric/AssignAddVariableOp�(model_1/add_metric/AssignAddVariableOp_1�,model_1/add_metric/div_no_nan/ReadVariableOp�.model_1/add_metric/div_no_nan/ReadVariableOp_1�(model_1/add_metric_1/AssignAddVariableOp�*model_1/add_metric_1/AssignAddVariableOp_1�.model_1/add_metric_1/div_no_nan/ReadVariableOp�0model_1/add_metric_1/div_no_nan/ReadVariableOp_1�+model_1/model/conv3d/BiasAdd/ReadVariableOp�-model_1/model/conv3d/BiasAdd_1/ReadVariableOp�*model_1/model/conv3d/Conv3D/ReadVariableOp�,model_1/model/conv3d/Conv3D_1/ReadVariableOp�-model_1/model/conv3d_1/BiasAdd/ReadVariableOp�/model_1/model/conv3d_1/BiasAdd_1/ReadVariableOp�,model_1/model/conv3d_1/Conv3D/ReadVariableOp�.model_1/model/conv3d_1/Conv3D_1/ReadVariableOp�-model_1/model/conv3d_2/BiasAdd/ReadVariableOp�/model_1/model/conv3d_2/BiasAdd_1/ReadVariableOp�,model_1/model/conv3d_2/Conv3D/ReadVariableOp�.model_1/model/conv3d_2/Conv3D_1/ReadVariableOp�-model_1/model/conv3d_3/BiasAdd/ReadVariableOp�/model_1/model/conv3d_3/BiasAdd_1/ReadVariableOp�,model_1/model/conv3d_3/Conv3D/ReadVariableOp�.model_1/model/conv3d_3/Conv3D_1/ReadVariableOp�-model_1/model/conv3d_4/BiasAdd/ReadVariableOp�/model_1/model/conv3d_4/BiasAdd_1/ReadVariableOp�,model_1/model/conv3d_4/Conv3D/ReadVariableOp�.model_1/model/conv3d_4/Conv3D_1/ReadVariableOp�-model_1/model/conv3d_5/BiasAdd/ReadVariableOp�/model_1/model/conv3d_5/BiasAdd_1/ReadVariableOp�,model_1/model/conv3d_5/Conv3D/ReadVariableOp�.model_1/model/conv3d_5/Conv3D_1/ReadVariableOp�-model_1/model/conv3d_6/BiasAdd/ReadVariableOp�/model_1/model/conv3d_6/BiasAdd_1/ReadVariableOp�,model_1/model/conv3d_6/Conv3D/ReadVariableOp�.model_1/model/conv3d_6/Conv3D_1/ReadVariableOp�-model_1/model/conv3d_7/BiasAdd/ReadVariableOp�/model_1/model/conv3d_7/BiasAdd_1/ReadVariableOp�,model_1/model/conv3d_7/Conv3D/ReadVariableOp�.model_1/model/conv3d_7/Conv3D_1/ReadVariableOp�*model_1/model/dense/BiasAdd/ReadVariableOp�,model_1/model/dense/BiasAdd_1/ReadVariableOp�)model_1/model/dense/MatMul/ReadVariableOp�+model_1/model/dense/MatMul_1/ReadVariableOp�
*model_1/model/conv3d/Conv3D/ReadVariableOpReadVariableOp3model_1_model_conv3d_conv3d_readvariableop_resource**
_output_shapes
:@*
dtype0�
model_1/model/conv3d/Conv3DConv3Dinput_32model_1/model/conv3d/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������22@*
paddingSAME*
strides	
�
+model_1/model/conv3d/BiasAdd/ReadVariableOpReadVariableOp4model_1_model_conv3d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
model_1/model/conv3d/BiasAddBiasAdd$model_1/model/conv3d/Conv3D:output:03model_1/model/conv3d/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������22@�
model_1/model/conv3d/ReluRelu%model_1/model/conv3d/BiasAdd:output:0*
T0*3
_output_shapes!
:���������22@�
,model_1/model/conv3d_1/Conv3D/ReadVariableOpReadVariableOp5model_1_model_conv3d_1_conv3d_readvariableop_resource**
_output_shapes
:@@*
dtype0�
model_1/model/conv3d_1/Conv3DConv3D'model_1/model/conv3d/Relu:activations:04model_1/model/conv3d_1/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������22@*
paddingSAME*
strides	
�
-model_1/model/conv3d_1/BiasAdd/ReadVariableOpReadVariableOp6model_1_model_conv3d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
model_1/model/conv3d_1/BiasAddBiasAdd&model_1/model/conv3d_1/Conv3D:output:05model_1/model/conv3d_1/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������22@�
model_1/model/conv3d_1/ReluRelu'model_1/model/conv3d_1/BiasAdd:output:0*
T0*3
_output_shapes!
:���������22@�
%model_1/model/max_pooling3d/MaxPool3D	MaxPool3D)model_1/model/conv3d_1/Relu:activations:0*
T0*3
_output_shapes!
:���������
@*
ksize	
*
paddingVALID*
strides	
�
,model_1/model/conv3d_2/Conv3D/ReadVariableOpReadVariableOp5model_1_model_conv3d_2_conv3d_readvariableop_resource*+
_output_shapes
:@�*
dtype0�
model_1/model/conv3d_2/Conv3DConv3D.model_1/model/max_pooling3d/MaxPool3D:output:04model_1/model/conv3d_2/Conv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :���������
�*
paddingSAME*
strides	
�
-model_1/model/conv3d_2/BiasAdd/ReadVariableOpReadVariableOp6model_1_model_conv3d_2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
model_1/model/conv3d_2/BiasAddBiasAdd&model_1/model/conv3d_2/Conv3D:output:05model_1/model/conv3d_2/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :���������
��
model_1/model/conv3d_2/ReluRelu'model_1/model/conv3d_2/BiasAdd:output:0*
T0*4
_output_shapes"
 :���������
��
,model_1/model/conv3d_3/Conv3D/ReadVariableOpReadVariableOp5model_1_model_conv3d_3_conv3d_readvariableop_resource*,
_output_shapes
:��*
dtype0�
model_1/model/conv3d_3/Conv3DConv3D)model_1/model/conv3d_2/Relu:activations:04model_1/model/conv3d_3/Conv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :���������
�*
paddingSAME*
strides	
�
-model_1/model/conv3d_3/BiasAdd/ReadVariableOpReadVariableOp6model_1_model_conv3d_3_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
model_1/model/conv3d_3/BiasAddBiasAdd&model_1/model/conv3d_3/Conv3D:output:05model_1/model/conv3d_3/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :���������
��
model_1/model/conv3d_3/ReluRelu'model_1/model/conv3d_3/BiasAdd:output:0*
T0*4
_output_shapes"
 :���������
��
'model_1/model/max_pooling3d_1/MaxPool3D	MaxPool3D)model_1/model/conv3d_3/Relu:activations:0*
T0*4
_output_shapes"
 :����������*
ksize	
*
paddingVALID*
strides	
�
,model_1/model/conv3d_4/Conv3D/ReadVariableOpReadVariableOp5model_1_model_conv3d_4_conv3d_readvariableop_resource*,
_output_shapes
:��*
dtype0�
model_1/model/conv3d_4/Conv3DConv3D0model_1/model/max_pooling3d_1/MaxPool3D:output:04model_1/model/conv3d_4/Conv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :����������*
paddingSAME*
strides	
�
-model_1/model/conv3d_4/BiasAdd/ReadVariableOpReadVariableOp6model_1_model_conv3d_4_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
model_1/model/conv3d_4/BiasAddBiasAdd&model_1/model/conv3d_4/Conv3D:output:05model_1/model/conv3d_4/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :�����������
model_1/model/conv3d_4/ReluRelu'model_1/model/conv3d_4/BiasAdd:output:0*
T0*4
_output_shapes"
 :�����������
,model_1/model/conv3d_5/Conv3D/ReadVariableOpReadVariableOp5model_1_model_conv3d_5_conv3d_readvariableop_resource*,
_output_shapes
:��*
dtype0�
model_1/model/conv3d_5/Conv3DConv3D)model_1/model/conv3d_4/Relu:activations:04model_1/model/conv3d_5/Conv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :����������*
paddingSAME*
strides	
�
-model_1/model/conv3d_5/BiasAdd/ReadVariableOpReadVariableOp6model_1_model_conv3d_5_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
model_1/model/conv3d_5/BiasAddBiasAdd&model_1/model/conv3d_5/Conv3D:output:05model_1/model/conv3d_5/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :�����������
model_1/model/conv3d_5/ReluRelu'model_1/model/conv3d_5/BiasAdd:output:0*
T0*4
_output_shapes"
 :�����������
'model_1/model/max_pooling3d_2/MaxPool3D	MaxPool3D)model_1/model/conv3d_5/Relu:activations:0*
T0*4
_output_shapes"
 :����������*
ksize	
*
paddingVALID*
strides	
�
,model_1/model/conv3d_6/Conv3D/ReadVariableOpReadVariableOp5model_1_model_conv3d_6_conv3d_readvariableop_resource*,
_output_shapes
:��*
dtype0�
model_1/model/conv3d_6/Conv3DConv3D0model_1/model/max_pooling3d_2/MaxPool3D:output:04model_1/model/conv3d_6/Conv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :����������*
paddingSAME*
strides	
�
-model_1/model/conv3d_6/BiasAdd/ReadVariableOpReadVariableOp6model_1_model_conv3d_6_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
model_1/model/conv3d_6/BiasAddBiasAdd&model_1/model/conv3d_6/Conv3D:output:05model_1/model/conv3d_6/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :�����������
model_1/model/conv3d_6/ReluRelu'model_1/model/conv3d_6/BiasAdd:output:0*
T0*4
_output_shapes"
 :�����������
,model_1/model/conv3d_7/Conv3D/ReadVariableOpReadVariableOp5model_1_model_conv3d_7_conv3d_readvariableop_resource*,
_output_shapes
:��*
dtype0�
model_1/model/conv3d_7/Conv3DConv3D)model_1/model/conv3d_6/Relu:activations:04model_1/model/conv3d_7/Conv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :����������*
paddingSAME*
strides	
�
-model_1/model/conv3d_7/BiasAdd/ReadVariableOpReadVariableOp6model_1_model_conv3d_7_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
model_1/model/conv3d_7/BiasAddBiasAdd&model_1/model/conv3d_7/Conv3D:output:05model_1/model/conv3d_7/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :�����������
model_1/model/conv3d_7/ReluRelu'model_1/model/conv3d_7/BiasAdd:output:0*
T0*4
_output_shapes"
 :�����������
'model_1/model/max_pooling3d_3/MaxPool3D	MaxPool3D)model_1/model/conv3d_7/Relu:activations:0*
T0*4
_output_shapes"
 :����������*
ksize	
*
paddingVALID*
strides	
l
model_1/model/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   �
model_1/model/flatten/ReshapeReshape0model_1/model/max_pooling3d_3/MaxPool3D:output:0$model_1/model/flatten/Const:output:0*
T0*(
_output_shapes
:����������$�
)model_1/model/dense/MatMul/ReadVariableOpReadVariableOp2model_1_model_dense_matmul_readvariableop_resource*
_output_shapes
:	�$*
dtype0�
model_1/model/dense/MatMulMatMul&model_1/model/flatten/Reshape:output:01model_1/model/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
*model_1/model/dense/BiasAdd/ReadVariableOpReadVariableOp3model_1_model_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model_1/model/dense/BiasAddBiasAdd$model_1/model/dense/MatMul:product:02model_1/model/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
,model_1/model/conv3d/Conv3D_1/ReadVariableOpReadVariableOp3model_1_model_conv3d_conv3d_readvariableop_resource**
_output_shapes
:@*
dtype0�
model_1/model/conv3d/Conv3D_1Conv3Dinput_24model_1/model/conv3d/Conv3D_1/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������22@*
paddingSAME*
strides	
�
-model_1/model/conv3d/BiasAdd_1/ReadVariableOpReadVariableOp4model_1_model_conv3d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
model_1/model/conv3d/BiasAdd_1BiasAdd&model_1/model/conv3d/Conv3D_1:output:05model_1/model/conv3d/BiasAdd_1/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������22@�
model_1/model/conv3d/Relu_1Relu'model_1/model/conv3d/BiasAdd_1:output:0*
T0*3
_output_shapes!
:���������22@�
.model_1/model/conv3d_1/Conv3D_1/ReadVariableOpReadVariableOp5model_1_model_conv3d_1_conv3d_readvariableop_resource**
_output_shapes
:@@*
dtype0�
model_1/model/conv3d_1/Conv3D_1Conv3D)model_1/model/conv3d/Relu_1:activations:06model_1/model/conv3d_1/Conv3D_1/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������22@*
paddingSAME*
strides	
�
/model_1/model/conv3d_1/BiasAdd_1/ReadVariableOpReadVariableOp6model_1_model_conv3d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
 model_1/model/conv3d_1/BiasAdd_1BiasAdd(model_1/model/conv3d_1/Conv3D_1:output:07model_1/model/conv3d_1/BiasAdd_1/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������22@�
model_1/model/conv3d_1/Relu_1Relu)model_1/model/conv3d_1/BiasAdd_1:output:0*
T0*3
_output_shapes!
:���������22@�
'model_1/model/max_pooling3d/MaxPool3D_1	MaxPool3D+model_1/model/conv3d_1/Relu_1:activations:0*
T0*3
_output_shapes!
:���������
@*
ksize	
*
paddingVALID*
strides	
�
.model_1/model/conv3d_2/Conv3D_1/ReadVariableOpReadVariableOp5model_1_model_conv3d_2_conv3d_readvariableop_resource*+
_output_shapes
:@�*
dtype0�
model_1/model/conv3d_2/Conv3D_1Conv3D0model_1/model/max_pooling3d/MaxPool3D_1:output:06model_1/model/conv3d_2/Conv3D_1/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :���������
�*
paddingSAME*
strides	
�
/model_1/model/conv3d_2/BiasAdd_1/ReadVariableOpReadVariableOp6model_1_model_conv3d_2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
 model_1/model/conv3d_2/BiasAdd_1BiasAdd(model_1/model/conv3d_2/Conv3D_1:output:07model_1/model/conv3d_2/BiasAdd_1/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :���������
��
model_1/model/conv3d_2/Relu_1Relu)model_1/model/conv3d_2/BiasAdd_1:output:0*
T0*4
_output_shapes"
 :���������
��
.model_1/model/conv3d_3/Conv3D_1/ReadVariableOpReadVariableOp5model_1_model_conv3d_3_conv3d_readvariableop_resource*,
_output_shapes
:��*
dtype0�
model_1/model/conv3d_3/Conv3D_1Conv3D+model_1/model/conv3d_2/Relu_1:activations:06model_1/model/conv3d_3/Conv3D_1/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :���������
�*
paddingSAME*
strides	
�
/model_1/model/conv3d_3/BiasAdd_1/ReadVariableOpReadVariableOp6model_1_model_conv3d_3_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
 model_1/model/conv3d_3/BiasAdd_1BiasAdd(model_1/model/conv3d_3/Conv3D_1:output:07model_1/model/conv3d_3/BiasAdd_1/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :���������
��
model_1/model/conv3d_3/Relu_1Relu)model_1/model/conv3d_3/BiasAdd_1:output:0*
T0*4
_output_shapes"
 :���������
��
)model_1/model/max_pooling3d_1/MaxPool3D_1	MaxPool3D+model_1/model/conv3d_3/Relu_1:activations:0*
T0*4
_output_shapes"
 :����������*
ksize	
*
paddingVALID*
strides	
�
.model_1/model/conv3d_4/Conv3D_1/ReadVariableOpReadVariableOp5model_1_model_conv3d_4_conv3d_readvariableop_resource*,
_output_shapes
:��*
dtype0�
model_1/model/conv3d_4/Conv3D_1Conv3D2model_1/model/max_pooling3d_1/MaxPool3D_1:output:06model_1/model/conv3d_4/Conv3D_1/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :����������*
paddingSAME*
strides	
�
/model_1/model/conv3d_4/BiasAdd_1/ReadVariableOpReadVariableOp6model_1_model_conv3d_4_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
 model_1/model/conv3d_4/BiasAdd_1BiasAdd(model_1/model/conv3d_4/Conv3D_1:output:07model_1/model/conv3d_4/BiasAdd_1/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :�����������
model_1/model/conv3d_4/Relu_1Relu)model_1/model/conv3d_4/BiasAdd_1:output:0*
T0*4
_output_shapes"
 :�����������
.model_1/model/conv3d_5/Conv3D_1/ReadVariableOpReadVariableOp5model_1_model_conv3d_5_conv3d_readvariableop_resource*,
_output_shapes
:��*
dtype0�
model_1/model/conv3d_5/Conv3D_1Conv3D+model_1/model/conv3d_4/Relu_1:activations:06model_1/model/conv3d_5/Conv3D_1/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :����������*
paddingSAME*
strides	
�
/model_1/model/conv3d_5/BiasAdd_1/ReadVariableOpReadVariableOp6model_1_model_conv3d_5_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
 model_1/model/conv3d_5/BiasAdd_1BiasAdd(model_1/model/conv3d_5/Conv3D_1:output:07model_1/model/conv3d_5/BiasAdd_1/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :�����������
model_1/model/conv3d_5/Relu_1Relu)model_1/model/conv3d_5/BiasAdd_1:output:0*
T0*4
_output_shapes"
 :�����������
)model_1/model/max_pooling3d_2/MaxPool3D_1	MaxPool3D+model_1/model/conv3d_5/Relu_1:activations:0*
T0*4
_output_shapes"
 :����������*
ksize	
*
paddingVALID*
strides	
�
.model_1/model/conv3d_6/Conv3D_1/ReadVariableOpReadVariableOp5model_1_model_conv3d_6_conv3d_readvariableop_resource*,
_output_shapes
:��*
dtype0�
model_1/model/conv3d_6/Conv3D_1Conv3D2model_1/model/max_pooling3d_2/MaxPool3D_1:output:06model_1/model/conv3d_6/Conv3D_1/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :����������*
paddingSAME*
strides	
�
/model_1/model/conv3d_6/BiasAdd_1/ReadVariableOpReadVariableOp6model_1_model_conv3d_6_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
 model_1/model/conv3d_6/BiasAdd_1BiasAdd(model_1/model/conv3d_6/Conv3D_1:output:07model_1/model/conv3d_6/BiasAdd_1/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :�����������
model_1/model/conv3d_6/Relu_1Relu)model_1/model/conv3d_6/BiasAdd_1:output:0*
T0*4
_output_shapes"
 :�����������
.model_1/model/conv3d_7/Conv3D_1/ReadVariableOpReadVariableOp5model_1_model_conv3d_7_conv3d_readvariableop_resource*,
_output_shapes
:��*
dtype0�
model_1/model/conv3d_7/Conv3D_1Conv3D+model_1/model/conv3d_6/Relu_1:activations:06model_1/model/conv3d_7/Conv3D_1/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :����������*
paddingSAME*
strides	
�
/model_1/model/conv3d_7/BiasAdd_1/ReadVariableOpReadVariableOp6model_1_model_conv3d_7_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
 model_1/model/conv3d_7/BiasAdd_1BiasAdd(model_1/model/conv3d_7/Conv3D_1:output:07model_1/model/conv3d_7/BiasAdd_1/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :�����������
model_1/model/conv3d_7/Relu_1Relu)model_1/model/conv3d_7/BiasAdd_1:output:0*
T0*4
_output_shapes"
 :�����������
)model_1/model/max_pooling3d_3/MaxPool3D_1	MaxPool3D+model_1/model/conv3d_7/Relu_1:activations:0*
T0*4
_output_shapes"
 :����������*
ksize	
*
paddingVALID*
strides	
n
model_1/model/flatten/Const_1Const*
_output_shapes
:*
dtype0*
valueB"����   �
model_1/model/flatten/Reshape_1Reshape2model_1/model/max_pooling3d_3/MaxPool3D_1:output:0&model_1/model/flatten/Const_1:output:0*
T0*(
_output_shapes
:����������$�
+model_1/model/dense/MatMul_1/ReadVariableOpReadVariableOp2model_1_model_dense_matmul_readvariableop_resource*
_output_shapes
:	�$*
dtype0�
model_1/model/dense/MatMul_1MatMul(model_1/model/flatten/Reshape_1:output:03model_1/model/dense/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
,model_1/model/dense/BiasAdd_1/ReadVariableOpReadVariableOp3model_1_model_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model_1/model/dense/BiasAdd_1BiasAdd&model_1/model/dense/MatMul_1:product:04model_1/model/dense/BiasAdd_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������]
&model_1/tf.compat.v1.squeeze_1/SqueezeSqueezeinput_5*
T0*
_output_shapes
:z
&model_1/tf.compat.v1.squeeze_3/SqueezeSqueeze$model_1/model/dense/BiasAdd:output:0*
T0*
_output_shapes
:[
$model_1/tf.compat.v1.squeeze/SqueezeSqueezeinput_4*
T0*
_output_shapes
:|
&model_1/tf.compat.v1.squeeze_2/SqueezeSqueeze&model_1/model/dense/BiasAdd_1:output:0*
T0*
_output_shapes
:|
&model_1/tf.compat.v1.squeeze_4/SqueezeSqueeze&model_1/model/dense/BiasAdd_1:output:0*
T0*
_output_shapes
:z
&model_1/tf.compat.v1.squeeze_5/SqueezeSqueeze$model_1/model/dense/BiasAdd:output:0*
T0*
_output_shapes
:]
&model_1/tf.compat.v1.squeeze_7/SqueezeSqueezeinput_5*
T0*
_output_shapes
:z
&model_1/tf.compat.v1.squeeze_9/SqueezeSqueeze$model_1/model/dense/BiasAdd:output:0*
T0*
_output_shapes
:]
&model_1/tf.compat.v1.squeeze_6/SqueezeSqueezeinput_4*
T0*
_output_shapes
:|
&model_1/tf.compat.v1.squeeze_8/SqueezeSqueeze&model_1/model/dense/BiasAdd_1:output:0*
T0*
_output_shapes
:{
'model_1/tf.compat.v1.squeeze_13/SqueezeSqueeze$model_1/model/dense/BiasAdd:output:0*
T0*
_output_shapes
:^
'model_1/tf.compat.v1.squeeze_12/SqueezeSqueezeinput_5*
T0*
_output_shapes
:}
'model_1/tf.compat.v1.squeeze_11/SqueezeSqueeze&model_1/model/dense/BiasAdd_1:output:0*
T0*
_output_shapes
:^
'model_1/tf.compat.v1.squeeze_10/SqueezeSqueezeinput_4*
T0*
_output_shapes
:�
model_1/tf.math.subtract_1/SubSub/model_1/tf.compat.v1.squeeze_1/Squeeze:output:0/model_1/tf.compat.v1.squeeze_3/Squeeze:output:0*
T0*
_output_shapes
:�
model_1/tf.math.subtract/SubSub-model_1/tf.compat.v1.squeeze/Squeeze:output:0/model_1/tf.compat.v1.squeeze_2/Squeeze:output:0*
T0*
_output_shapes
:�
model_1/tf.math.subtract_2/SubSub/model_1/tf.compat.v1.squeeze_4/Squeeze:output:0/model_1/tf.compat.v1.squeeze_5/Squeeze:output:0*
T0*
_output_shapes
:�
model_1/tf.math.subtract_5/SubSub/model_1/tf.compat.v1.squeeze_7/Squeeze:output:0/model_1/tf.compat.v1.squeeze_9/Squeeze:output:0*
T0*
_output_shapes
:�
model_1/tf.math.subtract_4/SubSub/model_1/tf.compat.v1.squeeze_6/Squeeze:output:0/model_1/tf.compat.v1.squeeze_8/Squeeze:output:0*
T0*
_output_shapes
:}
"model_1/tf.math.reduce_mean_8/RankRank0model_1/tf.compat.v1.squeeze_13/Squeeze:output:0*
T0*
_output_shapes
: k
)model_1/tf.math.reduce_mean_8/range/startConst*
_output_shapes
: *
dtype0*
value	B : k
)model_1/tf.math.reduce_mean_8/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
#model_1/tf.math.reduce_mean_8/rangeRange2model_1/tf.math.reduce_mean_8/range/start:output:0+model_1/tf.math.reduce_mean_8/Rank:output:02model_1/tf.math.reduce_mean_8/range/delta:output:0*#
_output_shapes
:����������
"model_1/tf.math.reduce_mean_8/MeanMean0model_1/tf.compat.v1.squeeze_13/Squeeze:output:0,model_1/tf.math.reduce_mean_8/range:output:0*
T0*
_output_shapes
: }
"model_1/tf.math.reduce_mean_7/RankRank0model_1/tf.compat.v1.squeeze_12/Squeeze:output:0*
T0*
_output_shapes
: k
)model_1/tf.math.reduce_mean_7/range/startConst*
_output_shapes
: *
dtype0*
value	B : k
)model_1/tf.math.reduce_mean_7/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
#model_1/tf.math.reduce_mean_7/rangeRange2model_1/tf.math.reduce_mean_7/range/start:output:0+model_1/tf.math.reduce_mean_7/Rank:output:02model_1/tf.math.reduce_mean_7/range/delta:output:0*#
_output_shapes
:����������
"model_1/tf.math.reduce_mean_7/MeanMean0model_1/tf.compat.v1.squeeze_12/Squeeze:output:0,model_1/tf.math.reduce_mean_7/range:output:0*
T0*
_output_shapes
: }
"model_1/tf.math.reduce_mean_6/RankRank0model_1/tf.compat.v1.squeeze_11/Squeeze:output:0*
T0*
_output_shapes
: k
)model_1/tf.math.reduce_mean_6/range/startConst*
_output_shapes
: *
dtype0*
value	B : k
)model_1/tf.math.reduce_mean_6/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
#model_1/tf.math.reduce_mean_6/rangeRange2model_1/tf.math.reduce_mean_6/range/start:output:0+model_1/tf.math.reduce_mean_6/Rank:output:02model_1/tf.math.reduce_mean_6/range/delta:output:0*#
_output_shapes
:����������
"model_1/tf.math.reduce_mean_6/MeanMean0model_1/tf.compat.v1.squeeze_11/Squeeze:output:0,model_1/tf.math.reduce_mean_6/range:output:0*
T0*
_output_shapes
: }
"model_1/tf.math.reduce_mean_5/RankRank0model_1/tf.compat.v1.squeeze_10/Squeeze:output:0*
T0*
_output_shapes
: k
)model_1/tf.math.reduce_mean_5/range/startConst*
_output_shapes
: *
dtype0*
value	B : k
)model_1/tf.math.reduce_mean_5/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
#model_1/tf.math.reduce_mean_5/rangeRange2model_1/tf.math.reduce_mean_5/range/start:output:0+model_1/tf.math.reduce_mean_5/Rank:output:02model_1/tf.math.reduce_mean_5/range/delta:output:0*#
_output_shapes
:����������
"model_1/tf.math.reduce_mean_5/MeanMean0model_1/tf.compat.v1.squeeze_10/Squeeze:output:0,model_1/tf.math.reduce_mean_5/range:output:0*
T0*
_output_shapes
: p
model_1/tf.math.square_1/SquareSquare"model_1/tf.math.subtract_1/Sub:z:0*
T0*
_output_shapes
:l
model_1/tf.math.square/SquareSquare model_1/tf.math.subtract/Sub:z:0*
T0*
_output_shapes
:p
model_1/tf.math.square_2/SquareSquare"model_1/tf.math.subtract_2/Sub:z:0*
T0*
_output_shapes
:p
model_1/tf.math.square_4/SquareSquare"model_1/tf.math.subtract_5/Sub:z:0*
T0*
_output_shapes
:p
model_1/tf.math.square_3/SquareSquare"model_1/tf.math.subtract_4/Sub:z:0*
T0*
_output_shapes
:�
model_1/tf.math.subtract_10/SubSub0model_1/tf.compat.v1.squeeze_13/Squeeze:output:0+model_1/tf.math.reduce_mean_8/Mean:output:0*
T0*
_output_shapes
:�
model_1/tf.math.subtract_9/SubSub0model_1/tf.compat.v1.squeeze_12/Squeeze:output:0+model_1/tf.math.reduce_mean_7/Mean:output:0*
T0*
_output_shapes
:�
model_1/tf.math.subtract_7/SubSub0model_1/tf.compat.v1.squeeze_11/Squeeze:output:0+model_1/tf.math.reduce_mean_6/Mean:output:0*
T0*
_output_shapes
:�
model_1/tf.math.subtract_6/SubSub0model_1/tf.compat.v1.squeeze_10/Squeeze:output:0+model_1/tf.math.reduce_mean_5/Mean:output:0*
T0*
_output_shapes
:l
 model_1/tf.math.reduce_mean/RankRank!model_1/tf.math.square/Square:y:0*
T0*
_output_shapes
: i
'model_1/tf.math.reduce_mean/range/startConst*
_output_shapes
: *
dtype0*
value	B : i
'model_1/tf.math.reduce_mean/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
!model_1/tf.math.reduce_mean/rangeRange0model_1/tf.math.reduce_mean/range/start:output:0)model_1/tf.math.reduce_mean/Rank:output:00model_1/tf.math.reduce_mean/range/delta:output:0*#
_output_shapes
:����������
 model_1/tf.math.reduce_mean/MeanMean!model_1/tf.math.square/Square:y:0*model_1/tf.math.reduce_mean/range:output:0*
T0*
_output_shapes
: p
"model_1/tf.math.reduce_mean_1/RankRank#model_1/tf.math.square_1/Square:y:0*
T0*
_output_shapes
: k
)model_1/tf.math.reduce_mean_1/range/startConst*
_output_shapes
: *
dtype0*
value	B : k
)model_1/tf.math.reduce_mean_1/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
#model_1/tf.math.reduce_mean_1/rangeRange2model_1/tf.math.reduce_mean_1/range/start:output:0+model_1/tf.math.reduce_mean_1/Rank:output:02model_1/tf.math.reduce_mean_1/range/delta:output:0*#
_output_shapes
:����������
"model_1/tf.math.reduce_mean_1/MeanMean#model_1/tf.math.square_1/Square:y:0,model_1/tf.math.reduce_mean_1/range:output:0*
T0*
_output_shapes
: f
!model_1/tf.math.maximum/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *��:�
model_1/tf.math.maximum/MaximumMaximum#model_1/tf.math.square_2/Square:y:0*model_1/tf.math.maximum/Maximum/y:output:0*
T0*
_output_shapes
:p
"model_1/tf.math.reduce_mean_3/RankRank#model_1/tf.math.square_3/Square:y:0*
T0*
_output_shapes
: k
)model_1/tf.math.reduce_mean_3/range/startConst*
_output_shapes
: *
dtype0*
value	B : k
)model_1/tf.math.reduce_mean_3/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
#model_1/tf.math.reduce_mean_3/rangeRange2model_1/tf.math.reduce_mean_3/range/start:output:0+model_1/tf.math.reduce_mean_3/Rank:output:02model_1/tf.math.reduce_mean_3/range/delta:output:0*#
_output_shapes
:����������
"model_1/tf.math.reduce_mean_3/MeanMean#model_1/tf.math.square_3/Square:y:0,model_1/tf.math.reduce_mean_3/range:output:0*
T0*
_output_shapes
: p
"model_1/tf.math.reduce_mean_4/RankRank#model_1/tf.math.square_4/Square:y:0*
T0*
_output_shapes
: k
)model_1/tf.math.reduce_mean_4/range/startConst*
_output_shapes
: *
dtype0*
value	B : k
)model_1/tf.math.reduce_mean_4/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
#model_1/tf.math.reduce_mean_4/rangeRange2model_1/tf.math.reduce_mean_4/range/start:output:0+model_1/tf.math.reduce_mean_4/Rank:output:02model_1/tf.math.reduce_mean_4/range/delta:output:0*#
_output_shapes
:����������
"model_1/tf.math.reduce_mean_4/MeanMean#model_1/tf.math.square_4/Square:y:0,model_1/tf.math.reduce_mean_4/range:output:0*
T0*
_output_shapes
: �
model_1/tf.math.multiply_3/MulMul"model_1/tf.math.subtract_9/Sub:z:0#model_1/tf.math.subtract_10/Sub:z:0*
T0*
_output_shapes
:�
model_1/tf.math.multiply_1/MulMul"model_1/tf.math.subtract_6/Sub:z:0"model_1/tf.math.subtract_7/Sub:z:0*
T0*
_output_shapes
:q
model_1/tf.math.square_8/SquareSquare#model_1/tf.math.subtract_10/Sub:z:0*
T0*
_output_shapes
:p
model_1/tf.math.square_7/SquareSquare"model_1/tf.math.subtract_9/Sub:z:0*
T0*
_output_shapes
:p
model_1/tf.math.square_6/SquareSquare"model_1/tf.math.subtract_7/Sub:z:0*
T0*
_output_shapes
:p
model_1/tf.math.square_5/SquareSquare"model_1/tf.math.subtract_6/Sub:z:0*
T0*
_output_shapes
:�
"model_1/tf.__operators__.add/AddV2AddV2)model_1/tf.math.reduce_mean/Mean:output:0+model_1/tf.math.reduce_mean_1/Mean:output:0*
T0*
_output_shapes
: e
 model_1/tf.math.subtract_3/Sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *��:�
model_1/tf.math.subtract_3/SubSub#model_1/tf.math.maximum/Maximum:z:0)model_1/tf.math.subtract_3/Sub/y:output:0*
T0*
_output_shapes
:�
$model_1/tf.__operators__.add_2/AddV2AddV2+model_1/tf.math.reduce_mean_3/Mean:output:0+model_1/tf.math.reduce_mean_4/Mean:output:0*
T0*
_output_shapes
: n
!model_1/tf.math.reduce_sum_3/RankRank"model_1/tf.math.multiply_3/Mul:z:0*
T0*
_output_shapes
: j
(model_1/tf.math.reduce_sum_3/range/startConst*
_output_shapes
: *
dtype0*
value	B : j
(model_1/tf.math.reduce_sum_3/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
"model_1/tf.math.reduce_sum_3/rangeRange1model_1/tf.math.reduce_sum_3/range/start:output:0*model_1/tf.math.reduce_sum_3/Rank:output:01model_1/tf.math.reduce_sum_3/range/delta:output:0*#
_output_shapes
:����������
 model_1/tf.math.reduce_sum_3/SumSum"model_1/tf.math.multiply_3/Mul:z:0+model_1/tf.math.reduce_sum_3/range:output:0*
T0*
_output_shapes
: l
model_1/tf.math.reduce_sum/RankRank"model_1/tf.math.multiply_1/Mul:z:0*
T0*
_output_shapes
: h
&model_1/tf.math.reduce_sum/range/startConst*
_output_shapes
: *
dtype0*
value	B : h
&model_1/tf.math.reduce_sum/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
 model_1/tf.math.reduce_sum/rangeRange/model_1/tf.math.reduce_sum/range/start:output:0(model_1/tf.math.reduce_sum/Rank:output:0/model_1/tf.math.reduce_sum/range/delta:output:0*#
_output_shapes
:����������
model_1/tf.math.reduce_sum/SumSum"model_1/tf.math.multiply_1/Mul:z:0)model_1/tf.math.reduce_sum/range:output:0*
T0*
_output_shapes
: o
!model_1/tf.math.reduce_sum_4/RankRank#model_1/tf.math.square_7/Square:y:0*
T0*
_output_shapes
: j
(model_1/tf.math.reduce_sum_4/range/startConst*
_output_shapes
: *
dtype0*
value	B : j
(model_1/tf.math.reduce_sum_4/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
"model_1/tf.math.reduce_sum_4/rangeRange1model_1/tf.math.reduce_sum_4/range/start:output:0*model_1/tf.math.reduce_sum_4/Rank:output:01model_1/tf.math.reduce_sum_4/range/delta:output:0*#
_output_shapes
:����������
 model_1/tf.math.reduce_sum_4/SumSum#model_1/tf.math.square_7/Square:y:0+model_1/tf.math.reduce_sum_4/range:output:0*
T0*
_output_shapes
: o
!model_1/tf.math.reduce_sum_5/RankRank#model_1/tf.math.square_8/Square:y:0*
T0*
_output_shapes
: j
(model_1/tf.math.reduce_sum_5/range/startConst*
_output_shapes
: *
dtype0*
value	B : j
(model_1/tf.math.reduce_sum_5/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
"model_1/tf.math.reduce_sum_5/rangeRange1model_1/tf.math.reduce_sum_5/range/start:output:0*model_1/tf.math.reduce_sum_5/Rank:output:01model_1/tf.math.reduce_sum_5/range/delta:output:0*#
_output_shapes
:����������
 model_1/tf.math.reduce_sum_5/SumSum#model_1/tf.math.square_8/Square:y:0+model_1/tf.math.reduce_sum_5/range:output:0*
T0*
_output_shapes
: o
!model_1/tf.math.reduce_sum_1/RankRank#model_1/tf.math.square_5/Square:y:0*
T0*
_output_shapes
: j
(model_1/tf.math.reduce_sum_1/range/startConst*
_output_shapes
: *
dtype0*
value	B : j
(model_1/tf.math.reduce_sum_1/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
"model_1/tf.math.reduce_sum_1/rangeRange1model_1/tf.math.reduce_sum_1/range/start:output:0*model_1/tf.math.reduce_sum_1/Rank:output:01model_1/tf.math.reduce_sum_1/range/delta:output:0*#
_output_shapes
:����������
 model_1/tf.math.reduce_sum_1/SumSum#model_1/tf.math.square_5/Square:y:0+model_1/tf.math.reduce_sum_1/range:output:0*
T0*
_output_shapes
: o
!model_1/tf.math.reduce_sum_2/RankRank#model_1/tf.math.square_6/Square:y:0*
T0*
_output_shapes
: j
(model_1/tf.math.reduce_sum_2/range/startConst*
_output_shapes
: *
dtype0*
value	B : j
(model_1/tf.math.reduce_sum_2/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
"model_1/tf.math.reduce_sum_2/rangeRange1model_1/tf.math.reduce_sum_2/range/start:output:0*model_1/tf.math.reduce_sum_2/Rank:output:01model_1/tf.math.reduce_sum_2/range/delta:output:0*#
_output_shapes
:����������
 model_1/tf.math.reduce_sum_2/SumSum#model_1/tf.math.square_6/Square:y:0+model_1/tf.math.reduce_sum_2/range:output:0*
T0*
_output_shapes
: o
"model_1/tf.math.reduce_mean_2/RankRank"model_1/tf.math.subtract_3/Sub:z:0*
T0*
_output_shapes
: k
)model_1/tf.math.reduce_mean_2/range/startConst*
_output_shapes
: *
dtype0*
value	B : k
)model_1/tf.math.reduce_mean_2/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
#model_1/tf.math.reduce_mean_2/rangeRange2model_1/tf.math.reduce_mean_2/range/start:output:0+model_1/tf.math.reduce_mean_2/Rank:output:02model_1/tf.math.reduce_mean_2/range/delta:output:0*#
_output_shapes
:����������
"model_1/tf.math.reduce_mean_2/MeanMean"model_1/tf.math.subtract_3/Sub:z:0,model_1/tf.math.reduce_mean_2/range:output:0*
T0*
_output_shapes
: Y
model_1/add_metric/RankConst*
_output_shapes
: *
dtype0*
value	B : `
model_1/add_metric/range/startConst*
_output_shapes
: *
dtype0*
value	B : `
model_1/add_metric/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
model_1/add_metric/rangeRange'model_1/add_metric/range/start:output:0 model_1/add_metric/Rank:output:0'model_1/add_metric/range/delta:output:0*
_output_shapes
: �
model_1/add_metric/SumSum(model_1/tf.__operators__.add_2/AddV2:z:0!model_1/add_metric/range:output:0*
T0*
_output_shapes
: �
&model_1/add_metric/AssignAddVariableOpAssignAddVariableOp/model_1_add_metric_assignaddvariableop_resourcemodel_1/add_metric/Sum:output:0*
_output_shapes
 *
dtype0Y
model_1/add_metric/SizeConst*
_output_shapes
: *
dtype0*
value	B :q
model_1/add_metric/CastCast model_1/add_metric/Size:output:0*

DstT0*

SrcT0*
_output_shapes
: �
(model_1/add_metric/AssignAddVariableOp_1AssignAddVariableOp1model_1_add_metric_assignaddvariableop_1_resourcemodel_1/add_metric/Cast:y:0'^model_1/add_metric/AssignAddVariableOp*
_output_shapes
 *
dtype0�
,model_1/add_metric/div_no_nan/ReadVariableOpReadVariableOp/model_1_add_metric_assignaddvariableop_resource'^model_1/add_metric/AssignAddVariableOp)^model_1/add_metric/AssignAddVariableOp_1*
_output_shapes
: *
dtype0�
.model_1/add_metric/div_no_nan/ReadVariableOp_1ReadVariableOp1model_1_add_metric_assignaddvariableop_1_resource)^model_1/add_metric/AssignAddVariableOp_1*
_output_shapes
: *
dtype0�
model_1/add_metric/div_no_nanDivNoNan4model_1/add_metric/div_no_nan/ReadVariableOp:value:06model_1/add_metric/div_no_nan/ReadVariableOp_1:value:0*
T0*
_output_shapes
: k
model_1/add_metric/IdentityIdentity!model_1/add_metric/div_no_nan:z:0*
T0*
_output_shapes
: �
model_1/tf.math.multiply_4/MulMul)model_1/tf.math.reduce_sum_4/Sum:output:0)model_1/tf.math.reduce_sum_5/Sum:output:0*
T0*
_output_shapes
: �
model_1/tf.math.multiply_2/MulMul)model_1/tf.math.reduce_sum_1/Sum:output:0)model_1/tf.math.reduce_sum_2/Sum:output:0*
T0*
_output_shapes
: �
model_1/tf.math.multiply/MulMulmodel_1_13086+model_1/tf.math.reduce_mean_2/Mean:output:0*
T0*
_output_shapes
: h
model_1/tf.math.sqrt_1/SqrtSqrt"model_1/tf.math.multiply_4/Mul:z:0*
T0*
_output_shapes
: f
model_1/tf.math.sqrt/SqrtSqrt"model_1/tf.math.multiply_2/Mul:z:0*
T0*
_output_shapes
: �
$model_1/tf.__operators__.add_1/AddV2AddV2&model_1/tf.__operators__.add/AddV2:z:0 model_1/tf.math.multiply/Mul:z:0*
T0*
_output_shapes
: e
 model_1/tf.__operators__.add_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *��'7�
$model_1/tf.__operators__.add_4/AddV2AddV2model_1/tf.math.sqrt_1/Sqrt:y:0)model_1/tf.__operators__.add_4/y:output:0*
T0*
_output_shapes
: e
 model_1/tf.__operators__.add_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *��'7�
$model_1/tf.__operators__.add_3/AddV2AddV2model_1/tf.math.sqrt/Sqrt:y:0)model_1/tf.__operators__.add_3/y:output:0*
T0*
_output_shapes
: �
!model_1/tf.math.truediv_1/truedivRealDiv)model_1/tf.math.reduce_sum_3/Sum:output:0(model_1/tf.__operators__.add_4/AddV2:z:0*
T0*
_output_shapes
: �
model_1/tf.math.truediv/truedivRealDiv'model_1/tf.math.reduce_sum/Sum:output:0(model_1/tf.__operators__.add_3/AddV2:z:0*
T0*
_output_shapes
: }
model_1/tf.math.subtract_11/SubSubmodel_1_13098%model_1/tf.math.truediv_1/truediv:z:0*
T0*
_output_shapes
: z
model_1/tf.math.subtract_8/SubSubmodel_1_13101#model_1/tf.math.truediv/truediv:z:0*
T0*
_output_shapes
: ^
model_1/tf.math.pow/Pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
model_1/tf.math.pow/PowPow"model_1/tf.math.subtract_8/Sub:z:0"model_1/tf.math.pow/Pow/y:output:0*
T0*
_output_shapes
: `
model_1/tf.math.pow_1/Pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
model_1/tf.math.pow_1/PowPow#model_1/tf.math.subtract_11/Sub:z:0$model_1/tf.math.pow_1/Pow/y:output:0*
T0*
_output_shapes
: �
$model_1/tf.__operators__.add_5/AddV2AddV2model_1/tf.math.pow/Pow:z:0model_1/tf.math.pow_1/Pow:z:0*
T0*
_output_shapes
: [
model_1/add_metric_1/RankConst*
_output_shapes
: *
dtype0*
value	B : b
 model_1/add_metric_1/range/startConst*
_output_shapes
: *
dtype0*
value	B : b
 model_1/add_metric_1/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
model_1/add_metric_1/rangeRange)model_1/add_metric_1/range/start:output:0"model_1/add_metric_1/Rank:output:0)model_1/add_metric_1/range/delta:output:0*
_output_shapes
: �
model_1/add_metric_1/SumSum(model_1/tf.__operators__.add_5/AddV2:z:0#model_1/add_metric_1/range:output:0*
T0*
_output_shapes
: �
(model_1/add_metric_1/AssignAddVariableOpAssignAddVariableOp1model_1_add_metric_1_assignaddvariableop_resource!model_1/add_metric_1/Sum:output:0*
_output_shapes
 *
dtype0[
model_1/add_metric_1/SizeConst*
_output_shapes
: *
dtype0*
value	B :u
model_1/add_metric_1/CastCast"model_1/add_metric_1/Size:output:0*

DstT0*

SrcT0*
_output_shapes
: �
*model_1/add_metric_1/AssignAddVariableOp_1AssignAddVariableOp3model_1_add_metric_1_assignaddvariableop_1_resourcemodel_1/add_metric_1/Cast:y:0)^model_1/add_metric_1/AssignAddVariableOp*
_output_shapes
 *
dtype0�
.model_1/add_metric_1/div_no_nan/ReadVariableOpReadVariableOp1model_1_add_metric_1_assignaddvariableop_resource)^model_1/add_metric_1/AssignAddVariableOp+^model_1/add_metric_1/AssignAddVariableOp_1*
_output_shapes
: *
dtype0�
0model_1/add_metric_1/div_no_nan/ReadVariableOp_1ReadVariableOp3model_1_add_metric_1_assignaddvariableop_1_resource+^model_1/add_metric_1/AssignAddVariableOp_1*
_output_shapes
: *
dtype0�
model_1/add_metric_1/div_no_nanDivNoNan6model_1/add_metric_1/div_no_nan/ReadVariableOp:value:08model_1/add_metric_1/div_no_nan/ReadVariableOp_1:value:0*
T0*
_output_shapes
: o
model_1/add_metric_1/IdentityIdentity#model_1/add_metric_1/div_no_nan:z:0*
T0*
_output_shapes
: u
IdentityIdentity&model_1/model/dense/BiasAdd_1:output:0^NoOp*
T0*'
_output_shapes
:���������u

Identity_1Identity$model_1/model/dense/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp'^model_1/add_metric/AssignAddVariableOp)^model_1/add_metric/AssignAddVariableOp_1-^model_1/add_metric/div_no_nan/ReadVariableOp/^model_1/add_metric/div_no_nan/ReadVariableOp_1)^model_1/add_metric_1/AssignAddVariableOp+^model_1/add_metric_1/AssignAddVariableOp_1/^model_1/add_metric_1/div_no_nan/ReadVariableOp1^model_1/add_metric_1/div_no_nan/ReadVariableOp_1,^model_1/model/conv3d/BiasAdd/ReadVariableOp.^model_1/model/conv3d/BiasAdd_1/ReadVariableOp+^model_1/model/conv3d/Conv3D/ReadVariableOp-^model_1/model/conv3d/Conv3D_1/ReadVariableOp.^model_1/model/conv3d_1/BiasAdd/ReadVariableOp0^model_1/model/conv3d_1/BiasAdd_1/ReadVariableOp-^model_1/model/conv3d_1/Conv3D/ReadVariableOp/^model_1/model/conv3d_1/Conv3D_1/ReadVariableOp.^model_1/model/conv3d_2/BiasAdd/ReadVariableOp0^model_1/model/conv3d_2/BiasAdd_1/ReadVariableOp-^model_1/model/conv3d_2/Conv3D/ReadVariableOp/^model_1/model/conv3d_2/Conv3D_1/ReadVariableOp.^model_1/model/conv3d_3/BiasAdd/ReadVariableOp0^model_1/model/conv3d_3/BiasAdd_1/ReadVariableOp-^model_1/model/conv3d_3/Conv3D/ReadVariableOp/^model_1/model/conv3d_3/Conv3D_1/ReadVariableOp.^model_1/model/conv3d_4/BiasAdd/ReadVariableOp0^model_1/model/conv3d_4/BiasAdd_1/ReadVariableOp-^model_1/model/conv3d_4/Conv3D/ReadVariableOp/^model_1/model/conv3d_4/Conv3D_1/ReadVariableOp.^model_1/model/conv3d_5/BiasAdd/ReadVariableOp0^model_1/model/conv3d_5/BiasAdd_1/ReadVariableOp-^model_1/model/conv3d_5/Conv3D/ReadVariableOp/^model_1/model/conv3d_5/Conv3D_1/ReadVariableOp.^model_1/model/conv3d_6/BiasAdd/ReadVariableOp0^model_1/model/conv3d_6/BiasAdd_1/ReadVariableOp-^model_1/model/conv3d_6/Conv3D/ReadVariableOp/^model_1/model/conv3d_6/Conv3D_1/ReadVariableOp.^model_1/model/conv3d_7/BiasAdd/ReadVariableOp0^model_1/model/conv3d_7/BiasAdd_1/ReadVariableOp-^model_1/model/conv3d_7/Conv3D/ReadVariableOp/^model_1/model/conv3d_7/Conv3D_1/ReadVariableOp+^model_1/model/dense/BiasAdd/ReadVariableOp-^model_1/model/dense/BiasAdd_1/ReadVariableOp*^model_1/model/dense/MatMul/ReadVariableOp,^model_1/model/dense/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������22:���������22:���������:���������: : : : : : : : : : : : : : : : : : : : : : : : : 2P
&model_1/add_metric/AssignAddVariableOp&model_1/add_metric/AssignAddVariableOp2T
(model_1/add_metric/AssignAddVariableOp_1(model_1/add_metric/AssignAddVariableOp_12\
,model_1/add_metric/div_no_nan/ReadVariableOp,model_1/add_metric/div_no_nan/ReadVariableOp2`
.model_1/add_metric/div_no_nan/ReadVariableOp_1.model_1/add_metric/div_no_nan/ReadVariableOp_12T
(model_1/add_metric_1/AssignAddVariableOp(model_1/add_metric_1/AssignAddVariableOp2X
*model_1/add_metric_1/AssignAddVariableOp_1*model_1/add_metric_1/AssignAddVariableOp_12`
.model_1/add_metric_1/div_no_nan/ReadVariableOp.model_1/add_metric_1/div_no_nan/ReadVariableOp2d
0model_1/add_metric_1/div_no_nan/ReadVariableOp_10model_1/add_metric_1/div_no_nan/ReadVariableOp_12Z
+model_1/model/conv3d/BiasAdd/ReadVariableOp+model_1/model/conv3d/BiasAdd/ReadVariableOp2^
-model_1/model/conv3d/BiasAdd_1/ReadVariableOp-model_1/model/conv3d/BiasAdd_1/ReadVariableOp2X
*model_1/model/conv3d/Conv3D/ReadVariableOp*model_1/model/conv3d/Conv3D/ReadVariableOp2\
,model_1/model/conv3d/Conv3D_1/ReadVariableOp,model_1/model/conv3d/Conv3D_1/ReadVariableOp2^
-model_1/model/conv3d_1/BiasAdd/ReadVariableOp-model_1/model/conv3d_1/BiasAdd/ReadVariableOp2b
/model_1/model/conv3d_1/BiasAdd_1/ReadVariableOp/model_1/model/conv3d_1/BiasAdd_1/ReadVariableOp2\
,model_1/model/conv3d_1/Conv3D/ReadVariableOp,model_1/model/conv3d_1/Conv3D/ReadVariableOp2`
.model_1/model/conv3d_1/Conv3D_1/ReadVariableOp.model_1/model/conv3d_1/Conv3D_1/ReadVariableOp2^
-model_1/model/conv3d_2/BiasAdd/ReadVariableOp-model_1/model/conv3d_2/BiasAdd/ReadVariableOp2b
/model_1/model/conv3d_2/BiasAdd_1/ReadVariableOp/model_1/model/conv3d_2/BiasAdd_1/ReadVariableOp2\
,model_1/model/conv3d_2/Conv3D/ReadVariableOp,model_1/model/conv3d_2/Conv3D/ReadVariableOp2`
.model_1/model/conv3d_2/Conv3D_1/ReadVariableOp.model_1/model/conv3d_2/Conv3D_1/ReadVariableOp2^
-model_1/model/conv3d_3/BiasAdd/ReadVariableOp-model_1/model/conv3d_3/BiasAdd/ReadVariableOp2b
/model_1/model/conv3d_3/BiasAdd_1/ReadVariableOp/model_1/model/conv3d_3/BiasAdd_1/ReadVariableOp2\
,model_1/model/conv3d_3/Conv3D/ReadVariableOp,model_1/model/conv3d_3/Conv3D/ReadVariableOp2`
.model_1/model/conv3d_3/Conv3D_1/ReadVariableOp.model_1/model/conv3d_3/Conv3D_1/ReadVariableOp2^
-model_1/model/conv3d_4/BiasAdd/ReadVariableOp-model_1/model/conv3d_4/BiasAdd/ReadVariableOp2b
/model_1/model/conv3d_4/BiasAdd_1/ReadVariableOp/model_1/model/conv3d_4/BiasAdd_1/ReadVariableOp2\
,model_1/model/conv3d_4/Conv3D/ReadVariableOp,model_1/model/conv3d_4/Conv3D/ReadVariableOp2`
.model_1/model/conv3d_4/Conv3D_1/ReadVariableOp.model_1/model/conv3d_4/Conv3D_1/ReadVariableOp2^
-model_1/model/conv3d_5/BiasAdd/ReadVariableOp-model_1/model/conv3d_5/BiasAdd/ReadVariableOp2b
/model_1/model/conv3d_5/BiasAdd_1/ReadVariableOp/model_1/model/conv3d_5/BiasAdd_1/ReadVariableOp2\
,model_1/model/conv3d_5/Conv3D/ReadVariableOp,model_1/model/conv3d_5/Conv3D/ReadVariableOp2`
.model_1/model/conv3d_5/Conv3D_1/ReadVariableOp.model_1/model/conv3d_5/Conv3D_1/ReadVariableOp2^
-model_1/model/conv3d_6/BiasAdd/ReadVariableOp-model_1/model/conv3d_6/BiasAdd/ReadVariableOp2b
/model_1/model/conv3d_6/BiasAdd_1/ReadVariableOp/model_1/model/conv3d_6/BiasAdd_1/ReadVariableOp2\
,model_1/model/conv3d_6/Conv3D/ReadVariableOp,model_1/model/conv3d_6/Conv3D/ReadVariableOp2`
.model_1/model/conv3d_6/Conv3D_1/ReadVariableOp.model_1/model/conv3d_6/Conv3D_1/ReadVariableOp2^
-model_1/model/conv3d_7/BiasAdd/ReadVariableOp-model_1/model/conv3d_7/BiasAdd/ReadVariableOp2b
/model_1/model/conv3d_7/BiasAdd_1/ReadVariableOp/model_1/model/conv3d_7/BiasAdd_1/ReadVariableOp2\
,model_1/model/conv3d_7/Conv3D/ReadVariableOp,model_1/model/conv3d_7/Conv3D/ReadVariableOp2`
.model_1/model/conv3d_7/Conv3D_1/ReadVariableOp.model_1/model/conv3d_7/Conv3D_1/ReadVariableOp2X
*model_1/model/dense/BiasAdd/ReadVariableOp*model_1/model/dense/BiasAdd/ReadVariableOp2\
,model_1/model/dense/BiasAdd_1/ReadVariableOp,model_1/model/dense/BiasAdd_1/ReadVariableOp2V
)model_1/model/dense/MatMul/ReadVariableOp)model_1/model/dense/MatMul/ReadVariableOp2Z
+model_1/model/dense/MatMul_1/ReadVariableOp+model_1/model/dense/MatMul_1/ReadVariableOp:\ X
3
_output_shapes!
:���������22
!
_user_specified_name	input_2:\X
3
_output_shapes!
:���������22
!
_user_specified_name	input_3:PL
'
_output_shapes
:���������
!
_user_specified_name	input_4:PL
'
_output_shapes
:���������
!
_user_specified_name	input_5:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
d
H__inference_max_pooling3d_layer_call_and_return_conditional_losses_16938

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
�
�
&__inference_conv3d_layer_call_fn_16885

inputs%
unknown:@
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
:���������22@*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *J
fERC
A__inference_conv3d_layer_call_and_return_conditional_losses_13196{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:���������22@`
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
�
�
#__inference_signature_wrapper_15555
input_2
input_3
input_4
input_5%
unknown:@
	unknown_0:@'
	unknown_1:@@
	unknown_2:@(
	unknown_3:@�
	unknown_4:	�)
	unknown_5:��
	unknown_6:	�)
	unknown_7:��
	unknown_8:	�)
	unknown_9:��

unknown_10:	�*

unknown_11:��

unknown_12:	�*

unknown_13:��

unknown_14:	�

unknown_15:	�$

unknown_16:

unknown_17: 

unknown_18: 

unknown_19

unknown_20

unknown_21

unknown_22: 

unknown_23: 
identity

identity_1��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_2input_3input_4input_5unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23*(
Tin!
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:���������:���������*4
_read_only_resource_inputs
	
*4
config_proto$"

CPU

GPU(2*0J 8� *)
f$R"
 __inference__wrapped_model_13124o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������22:���������22:���������:���������: : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
3
_output_shapes!
:���������22
!
_user_specified_name	input_2:\X
3
_output_shapes!
:���������22
!
_user_specified_name	input_3:PL
'
_output_shapes
:���������
!
_user_specified_name	input_4:PL
'
_output_shapes
:���������
!
_user_specified_name	input_5:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
�
'__inference_model_1_layer_call_fn_14410
input_2
input_3
input_4
input_5%
unknown:@
	unknown_0:@'
	unknown_1:@@
	unknown_2:@(
	unknown_3:@�
	unknown_4:	�)
	unknown_5:��
	unknown_6:	�)
	unknown_7:��
	unknown_8:	�)
	unknown_9:��

unknown_10:	�*

unknown_11:��

unknown_12:	�*

unknown_13:��

unknown_14:	�

unknown_15:	�$

unknown_16:

unknown_17: 

unknown_18: 

unknown_19

unknown_20

unknown_21

unknown_22: 

unknown_23: 
identity

identity_1��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_2input_3input_4input_5unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23*(
Tin!
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:���������:���������: *4
_read_only_resource_inputs
	
*4
config_proto$"

CPU

GPU(2*0J 8� *K
fFRD
B__inference_model_1_layer_call_and_return_conditional_losses_14354o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������22:���������22:���������:���������: : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
3
_output_shapes!
:���������22
!
_user_specified_name	input_2:\X
3
_output_shapes!
:���������22
!
_user_specified_name	input_3:PL
'
_output_shapes
:���������
!
_user_specified_name	input_4:PL
'
_output_shapes
:���������
!
_user_specified_name	input_5:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
C
'__inference_flatten_layer_call_fn_17129

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
:����������$* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8� *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_13373a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������$"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������:\ X
4
_output_shapes"
 :����������
 
_user_specified_nameinputs
�
�
C__inference_conv3d_6_layer_call_and_return_conditional_losses_17088

inputs>
conv3d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv3D/ReadVariableOp�1conv3d_6/kernel/Regularizer/Square/ReadVariableOp�
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource*,
_output_shapes
:��*
dtype0�
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :����������*
paddingSAME*
strides	
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :����������]
ReluReluBiasAdd:output:0*
T0*4
_output_shapes"
 :�����������
1conv3d_6/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv3d_readvariableop_resource*,
_output_shapes
:��*
dtype0�
"conv3d_6/kernel/Regularizer/SquareSquare9conv3d_6/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*,
_output_shapes
:��~
!conv3d_6/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
conv3d_6/kernel/Regularizer/SumSum&conv3d_6/kernel/Regularizer/Square:y:0*conv3d_6/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!conv3d_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    �
conv3d_6/kernel/Regularizer/mulMul*conv3d_6/kernel/Regularizer/mul/x:output:0(conv3d_6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: n
IdentityIdentityRelu:activations:0^NoOp*
T0*4
_output_shapes"
 :�����������
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp2^conv3d_6/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp2f
1conv3d_6/kernel/Regularizer/Square/ReadVariableOp1conv3d_6/kernel/Regularizer/Square/ReadVariableOp:\ X
4
_output_shapes"
 :����������
 
_user_specified_nameinputs
�
^
B__inference_flatten_layer_call_and_return_conditional_losses_13373

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"����   ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������$Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������$"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������:\ X
4
_output_shapes"
 :����������
 
_user_specified_nameinputs
�
I
-__inference_max_pooling3d_layer_call_fn_16933

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
H__inference_max_pooling3d_layer_call_and_return_conditional_losses_13133�
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
�
�
'__inference_model_1_layer_call_fn_15731
inputs_0
inputs_1
inputs_2
inputs_3%
unknown:@
	unknown_0:@'
	unknown_1:@@
	unknown_2:@(
	unknown_3:@�
	unknown_4:	�)
	unknown_5:��
	unknown_6:	�)
	unknown_7:��
	unknown_8:	�)
	unknown_9:��

unknown_10:	�*

unknown_11:��

unknown_12:	�*

unknown_13:��

unknown_14:	�

unknown_15:	�$

unknown_16:

unknown_17: 

unknown_18: 

unknown_19

unknown_20

unknown_21

unknown_22: 

unknown_23: 
identity

identity_1��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1inputs_2inputs_3unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23*(
Tin!
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:���������:���������: *4
_read_only_resource_inputs
	
*4
config_proto$"

CPU

GPU(2*0J 8� *K
fFRD
B__inference_model_1_layer_call_and_return_conditional_losses_14774o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������22:���������22:���������:���������: : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
3
_output_shapes!
:���������22
"
_user_specified_name
inputs/0:]Y
3
_output_shapes!
:���������22
"
_user_specified_name
inputs/1:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs/2:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs/3:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
�
A__inference_conv3d_layer_call_and_return_conditional_losses_13196

inputs<
conv3d_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�Conv3D/ReadVariableOp�/conv3d/kernel/Regularizer/Square/ReadVariableOp�
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:@*
dtype0�
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������22@*
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
:���������22@\
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:���������22@�
/conv3d/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:@*
dtype0�
 conv3d/kernel/Regularizer/SquareSquare7conv3d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0**
_output_shapes
:@|
conv3d/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
conv3d/kernel/Regularizer/SumSum$conv3d/kernel/Regularizer/Square:y:0(conv3d/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: d
conv3d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    �
conv3d/kernel/Regularizer/mulMul(conv3d/kernel/Regularizer/mul/x:output:0&conv3d/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:���������22@�
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp0^conv3d/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������22: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp2b
/conv3d/kernel/Regularizer/Square/ReadVariableOp/conv3d/kernel/Regularizer/Square/ReadVariableOp:[ W
3
_output_shapes!
:���������22
 
_user_specified_nameinputs
�
�
(__inference_conv3d_5_layer_call_fn_17035

inputs'
unknown:��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :����������*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *L
fGRE
C__inference_conv3d_5_layer_call_and_return_conditional_losses_13313|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :����������
 
_user_specified_nameinputs
��
�
@__inference_model_layer_call_and_return_conditional_losses_16813

inputsC
%conv3d_conv3d_readvariableop_resource:@4
&conv3d_biasadd_readvariableop_resource:@E
'conv3d_1_conv3d_readvariableop_resource:@@6
(conv3d_1_biasadd_readvariableop_resource:@F
'conv3d_2_conv3d_readvariableop_resource:@�7
(conv3d_2_biasadd_readvariableop_resource:	�G
'conv3d_3_conv3d_readvariableop_resource:��7
(conv3d_3_biasadd_readvariableop_resource:	�G
'conv3d_4_conv3d_readvariableop_resource:��7
(conv3d_4_biasadd_readvariableop_resource:	�G
'conv3d_5_conv3d_readvariableop_resource:��7
(conv3d_5_biasadd_readvariableop_resource:	�G
'conv3d_6_conv3d_readvariableop_resource:��7
(conv3d_6_biasadd_readvariableop_resource:	�G
'conv3d_7_conv3d_readvariableop_resource:��7
(conv3d_7_biasadd_readvariableop_resource:	�7
$dense_matmul_readvariableop_resource:	�$3
%dense_biasadd_readvariableop_resource:
identity��conv3d/BiasAdd/ReadVariableOp�conv3d/Conv3D/ReadVariableOp�/conv3d/kernel/Regularizer/Square/ReadVariableOp�conv3d_1/BiasAdd/ReadVariableOp�conv3d_1/Conv3D/ReadVariableOp�1conv3d_1/kernel/Regularizer/Square/ReadVariableOp�conv3d_2/BiasAdd/ReadVariableOp�conv3d_2/Conv3D/ReadVariableOp�1conv3d_2/kernel/Regularizer/Square/ReadVariableOp�conv3d_3/BiasAdd/ReadVariableOp�conv3d_3/Conv3D/ReadVariableOp�1conv3d_3/kernel/Regularizer/Square/ReadVariableOp�conv3d_4/BiasAdd/ReadVariableOp�conv3d_4/Conv3D/ReadVariableOp�1conv3d_4/kernel/Regularizer/Square/ReadVariableOp�conv3d_5/BiasAdd/ReadVariableOp�conv3d_5/Conv3D/ReadVariableOp�1conv3d_5/kernel/Regularizer/Square/ReadVariableOp�conv3d_6/BiasAdd/ReadVariableOp�conv3d_6/Conv3D/ReadVariableOp�1conv3d_6/kernel/Regularizer/Square/ReadVariableOp�conv3d_7/BiasAdd/ReadVariableOp�conv3d_7/Conv3D/ReadVariableOp�1conv3d_7/kernel/Regularizer/Square/ReadVariableOp�dense/BiasAdd/ReadVariableOp�dense/MatMul/ReadVariableOp�.dense/kernel/Regularizer/Square/ReadVariableOp�
conv3d/Conv3D/ReadVariableOpReadVariableOp%conv3d_conv3d_readvariableop_resource**
_output_shapes
:@*
dtype0�
conv3d/Conv3DConv3Dinputs$conv3d/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������22@*
paddingSAME*
strides	
�
conv3d/BiasAdd/ReadVariableOpReadVariableOp&conv3d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
conv3d/BiasAddBiasAddconv3d/Conv3D:output:0%conv3d/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������22@j
conv3d/ReluReluconv3d/BiasAdd:output:0*
T0*3
_output_shapes!
:���������22@�
conv3d_1/Conv3D/ReadVariableOpReadVariableOp'conv3d_1_conv3d_readvariableop_resource**
_output_shapes
:@@*
dtype0�
conv3d_1/Conv3DConv3Dconv3d/Relu:activations:0&conv3d_1/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������22@*
paddingSAME*
strides	
�
conv3d_1/BiasAdd/ReadVariableOpReadVariableOp(conv3d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
conv3d_1/BiasAddBiasAddconv3d_1/Conv3D:output:0'conv3d_1/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������22@n
conv3d_1/ReluReluconv3d_1/BiasAdd:output:0*
T0*3
_output_shapes!
:���������22@�
max_pooling3d/MaxPool3D	MaxPool3Dconv3d_1/Relu:activations:0*
T0*3
_output_shapes!
:���������
@*
ksize	
*
paddingVALID*
strides	
�
conv3d_2/Conv3D/ReadVariableOpReadVariableOp'conv3d_2_conv3d_readvariableop_resource*+
_output_shapes
:@�*
dtype0�
conv3d_2/Conv3DConv3D max_pooling3d/MaxPool3D:output:0&conv3d_2/Conv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :���������
�*
paddingSAME*
strides	
�
conv3d_2/BiasAdd/ReadVariableOpReadVariableOp(conv3d_2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
conv3d_2/BiasAddBiasAddconv3d_2/Conv3D:output:0'conv3d_2/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :���������
�o
conv3d_2/ReluReluconv3d_2/BiasAdd:output:0*
T0*4
_output_shapes"
 :���������
��
conv3d_3/Conv3D/ReadVariableOpReadVariableOp'conv3d_3_conv3d_readvariableop_resource*,
_output_shapes
:��*
dtype0�
conv3d_3/Conv3DConv3Dconv3d_2/Relu:activations:0&conv3d_3/Conv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :���������
�*
paddingSAME*
strides	
�
conv3d_3/BiasAdd/ReadVariableOpReadVariableOp(conv3d_3_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
conv3d_3/BiasAddBiasAddconv3d_3/Conv3D:output:0'conv3d_3/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :���������
�o
conv3d_3/ReluReluconv3d_3/BiasAdd:output:0*
T0*4
_output_shapes"
 :���������
��
max_pooling3d_1/MaxPool3D	MaxPool3Dconv3d_3/Relu:activations:0*
T0*4
_output_shapes"
 :����������*
ksize	
*
paddingVALID*
strides	
�
conv3d_4/Conv3D/ReadVariableOpReadVariableOp'conv3d_4_conv3d_readvariableop_resource*,
_output_shapes
:��*
dtype0�
conv3d_4/Conv3DConv3D"max_pooling3d_1/MaxPool3D:output:0&conv3d_4/Conv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :����������*
paddingSAME*
strides	
�
conv3d_4/BiasAdd/ReadVariableOpReadVariableOp(conv3d_4_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
conv3d_4/BiasAddBiasAddconv3d_4/Conv3D:output:0'conv3d_4/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :����������o
conv3d_4/ReluReluconv3d_4/BiasAdd:output:0*
T0*4
_output_shapes"
 :�����������
conv3d_5/Conv3D/ReadVariableOpReadVariableOp'conv3d_5_conv3d_readvariableop_resource*,
_output_shapes
:��*
dtype0�
conv3d_5/Conv3DConv3Dconv3d_4/Relu:activations:0&conv3d_5/Conv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :����������*
paddingSAME*
strides	
�
conv3d_5/BiasAdd/ReadVariableOpReadVariableOp(conv3d_5_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
conv3d_5/BiasAddBiasAddconv3d_5/Conv3D:output:0'conv3d_5/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :����������o
conv3d_5/ReluReluconv3d_5/BiasAdd:output:0*
T0*4
_output_shapes"
 :�����������
max_pooling3d_2/MaxPool3D	MaxPool3Dconv3d_5/Relu:activations:0*
T0*4
_output_shapes"
 :����������*
ksize	
*
paddingVALID*
strides	
�
conv3d_6/Conv3D/ReadVariableOpReadVariableOp'conv3d_6_conv3d_readvariableop_resource*,
_output_shapes
:��*
dtype0�
conv3d_6/Conv3DConv3D"max_pooling3d_2/MaxPool3D:output:0&conv3d_6/Conv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :����������*
paddingSAME*
strides	
�
conv3d_6/BiasAdd/ReadVariableOpReadVariableOp(conv3d_6_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
conv3d_6/BiasAddBiasAddconv3d_6/Conv3D:output:0'conv3d_6/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :����������o
conv3d_6/ReluReluconv3d_6/BiasAdd:output:0*
T0*4
_output_shapes"
 :�����������
conv3d_7/Conv3D/ReadVariableOpReadVariableOp'conv3d_7_conv3d_readvariableop_resource*,
_output_shapes
:��*
dtype0�
conv3d_7/Conv3DConv3Dconv3d_6/Relu:activations:0&conv3d_7/Conv3D/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :����������*
paddingSAME*
strides	
�
conv3d_7/BiasAdd/ReadVariableOpReadVariableOp(conv3d_7_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
conv3d_7/BiasAddBiasAddconv3d_7/Conv3D:output:0'conv3d_7/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :����������o
conv3d_7/ReluReluconv3d_7/BiasAdd:output:0*
T0*4
_output_shapes"
 :�����������
max_pooling3d_3/MaxPool3D	MaxPool3Dconv3d_7/Relu:activations:0*
T0*4
_output_shapes"
 :����������*
ksize	
*
paddingVALID*
strides	
^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   �
flatten/ReshapeReshape"max_pooling3d_3/MaxPool3D:output:0flatten/Const:output:0*
T0*(
_output_shapes
:����������$�
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	�$*
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
/conv3d/kernel/Regularizer/Square/ReadVariableOpReadVariableOp%conv3d_conv3d_readvariableop_resource**
_output_shapes
:@*
dtype0�
 conv3d/kernel/Regularizer/SquareSquare7conv3d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0**
_output_shapes
:@|
conv3d/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
conv3d/kernel/Regularizer/SumSum$conv3d/kernel/Regularizer/Square:y:0(conv3d/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: d
conv3d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    �
conv3d/kernel/Regularizer/mulMul(conv3d/kernel/Regularizer/mul/x:output:0&conv3d/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1conv3d_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv3d_1_conv3d_readvariableop_resource**
_output_shapes
:@@*
dtype0�
"conv3d_1/kernel/Regularizer/SquareSquare9conv3d_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0**
_output_shapes
:@@~
!conv3d_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
conv3d_1/kernel/Regularizer/SumSum&conv3d_1/kernel/Regularizer/Square:y:0*conv3d_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!conv3d_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    �
conv3d_1/kernel/Regularizer/mulMul*conv3d_1/kernel/Regularizer/mul/x:output:0(conv3d_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1conv3d_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv3d_2_conv3d_readvariableop_resource*+
_output_shapes
:@�*
dtype0�
"conv3d_2/kernel/Regularizer/SquareSquare9conv3d_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*+
_output_shapes
:@�~
!conv3d_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
conv3d_2/kernel/Regularizer/SumSum&conv3d_2/kernel/Regularizer/Square:y:0*conv3d_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!conv3d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    �
conv3d_2/kernel/Regularizer/mulMul*conv3d_2/kernel/Regularizer/mul/x:output:0(conv3d_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1conv3d_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv3d_3_conv3d_readvariableop_resource*,
_output_shapes
:��*
dtype0�
"conv3d_3/kernel/Regularizer/SquareSquare9conv3d_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*,
_output_shapes
:��~
!conv3d_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
conv3d_3/kernel/Regularizer/SumSum&conv3d_3/kernel/Regularizer/Square:y:0*conv3d_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!conv3d_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    �
conv3d_3/kernel/Regularizer/mulMul*conv3d_3/kernel/Regularizer/mul/x:output:0(conv3d_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1conv3d_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv3d_4_conv3d_readvariableop_resource*,
_output_shapes
:��*
dtype0�
"conv3d_4/kernel/Regularizer/SquareSquare9conv3d_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*,
_output_shapes
:��~
!conv3d_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
conv3d_4/kernel/Regularizer/SumSum&conv3d_4/kernel/Regularizer/Square:y:0*conv3d_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!conv3d_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    �
conv3d_4/kernel/Regularizer/mulMul*conv3d_4/kernel/Regularizer/mul/x:output:0(conv3d_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1conv3d_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv3d_5_conv3d_readvariableop_resource*,
_output_shapes
:��*
dtype0�
"conv3d_5/kernel/Regularizer/SquareSquare9conv3d_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*,
_output_shapes
:��~
!conv3d_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
conv3d_5/kernel/Regularizer/SumSum&conv3d_5/kernel/Regularizer/Square:y:0*conv3d_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!conv3d_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    �
conv3d_5/kernel/Regularizer/mulMul*conv3d_5/kernel/Regularizer/mul/x:output:0(conv3d_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1conv3d_6/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv3d_6_conv3d_readvariableop_resource*,
_output_shapes
:��*
dtype0�
"conv3d_6/kernel/Regularizer/SquareSquare9conv3d_6/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*,
_output_shapes
:��~
!conv3d_6/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
conv3d_6/kernel/Regularizer/SumSum&conv3d_6/kernel/Regularizer/Square:y:0*conv3d_6/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!conv3d_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    �
conv3d_6/kernel/Regularizer/mulMul*conv3d_6/kernel/Regularizer/mul/x:output:0(conv3d_6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1conv3d_7/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv3d_7_conv3d_readvariableop_resource*,
_output_shapes
:��*
dtype0�
"conv3d_7/kernel/Regularizer/SquareSquare9conv3d_7/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*,
_output_shapes
:��~
!conv3d_7/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
conv3d_7/kernel/Regularizer/SumSum&conv3d_7/kernel/Regularizer/Square:y:0*conv3d_7/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!conv3d_7/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    �
conv3d_7/kernel/Regularizer/mulMul*conv3d_7/kernel/Regularizer/mul/x:output:0(conv3d_7/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	�$*
dtype0�
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	�$o
dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
dense/kernel/Regularizer/SumSum#dense/kernel/Regularizer/Square:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: c
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    �
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: e
IdentityIdentitydense/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^conv3d/BiasAdd/ReadVariableOp^conv3d/Conv3D/ReadVariableOp0^conv3d/kernel/Regularizer/Square/ReadVariableOp ^conv3d_1/BiasAdd/ReadVariableOp^conv3d_1/Conv3D/ReadVariableOp2^conv3d_1/kernel/Regularizer/Square/ReadVariableOp ^conv3d_2/BiasAdd/ReadVariableOp^conv3d_2/Conv3D/ReadVariableOp2^conv3d_2/kernel/Regularizer/Square/ReadVariableOp ^conv3d_3/BiasAdd/ReadVariableOp^conv3d_3/Conv3D/ReadVariableOp2^conv3d_3/kernel/Regularizer/Square/ReadVariableOp ^conv3d_4/BiasAdd/ReadVariableOp^conv3d_4/Conv3D/ReadVariableOp2^conv3d_4/kernel/Regularizer/Square/ReadVariableOp ^conv3d_5/BiasAdd/ReadVariableOp^conv3d_5/Conv3D/ReadVariableOp2^conv3d_5/kernel/Regularizer/Square/ReadVariableOp ^conv3d_6/BiasAdd/ReadVariableOp^conv3d_6/Conv3D/ReadVariableOp2^conv3d_6/kernel/Regularizer/Square/ReadVariableOp ^conv3d_7/BiasAdd/ReadVariableOp^conv3d_7/Conv3D/ReadVariableOp2^conv3d_7/kernel/Regularizer/Square/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp/^dense/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:���������22: : : : : : : : : : : : : : : : : : 2>
conv3d/BiasAdd/ReadVariableOpconv3d/BiasAdd/ReadVariableOp2<
conv3d/Conv3D/ReadVariableOpconv3d/Conv3D/ReadVariableOp2b
/conv3d/kernel/Regularizer/Square/ReadVariableOp/conv3d/kernel/Regularizer/Square/ReadVariableOp2B
conv3d_1/BiasAdd/ReadVariableOpconv3d_1/BiasAdd/ReadVariableOp2@
conv3d_1/Conv3D/ReadVariableOpconv3d_1/Conv3D/ReadVariableOp2f
1conv3d_1/kernel/Regularizer/Square/ReadVariableOp1conv3d_1/kernel/Regularizer/Square/ReadVariableOp2B
conv3d_2/BiasAdd/ReadVariableOpconv3d_2/BiasAdd/ReadVariableOp2@
conv3d_2/Conv3D/ReadVariableOpconv3d_2/Conv3D/ReadVariableOp2f
1conv3d_2/kernel/Regularizer/Square/ReadVariableOp1conv3d_2/kernel/Regularizer/Square/ReadVariableOp2B
conv3d_3/BiasAdd/ReadVariableOpconv3d_3/BiasAdd/ReadVariableOp2@
conv3d_3/Conv3D/ReadVariableOpconv3d_3/Conv3D/ReadVariableOp2f
1conv3d_3/kernel/Regularizer/Square/ReadVariableOp1conv3d_3/kernel/Regularizer/Square/ReadVariableOp2B
conv3d_4/BiasAdd/ReadVariableOpconv3d_4/BiasAdd/ReadVariableOp2@
conv3d_4/Conv3D/ReadVariableOpconv3d_4/Conv3D/ReadVariableOp2f
1conv3d_4/kernel/Regularizer/Square/ReadVariableOp1conv3d_4/kernel/Regularizer/Square/ReadVariableOp2B
conv3d_5/BiasAdd/ReadVariableOpconv3d_5/BiasAdd/ReadVariableOp2@
conv3d_5/Conv3D/ReadVariableOpconv3d_5/Conv3D/ReadVariableOp2f
1conv3d_5/kernel/Regularizer/Square/ReadVariableOp1conv3d_5/kernel/Regularizer/Square/ReadVariableOp2B
conv3d_6/BiasAdd/ReadVariableOpconv3d_6/BiasAdd/ReadVariableOp2@
conv3d_6/Conv3D/ReadVariableOpconv3d_6/Conv3D/ReadVariableOp2f
1conv3d_6/kernel/Regularizer/Square/ReadVariableOp1conv3d_6/kernel/Regularizer/Square/ReadVariableOp2B
conv3d_7/BiasAdd/ReadVariableOpconv3d_7/BiasAdd/ReadVariableOp2@
conv3d_7/Conv3D/ReadVariableOpconv3d_7/Conv3D/ReadVariableOp2f
1conv3d_7/kernel/Regularizer/Square/ReadVariableOp1conv3d_7/kernel/Regularizer/Square/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2`
.dense/kernel/Regularizer/Square/ReadVariableOp.dense/kernel/Regularizer/Square/ReadVariableOp:[ W
3
_output_shapes!
:���������22
 
_user_specified_nameinputs
�
f
J__inference_max_pooling3d_2_layer_call_and_return_conditional_losses_13157

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
�
�
C__inference_conv3d_1_layer_call_and_return_conditional_losses_16928

inputs<
conv3d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�Conv3D/ReadVariableOp�1conv3d_1/kernel/Regularizer/Square/ReadVariableOp�
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:@@*
dtype0�
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������22@*
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
:���������22@\
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:���������22@�
1conv3d_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:@@*
dtype0�
"conv3d_1/kernel/Regularizer/SquareSquare9conv3d_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0**
_output_shapes
:@@~
!conv3d_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
conv3d_1/kernel/Regularizer/SumSum&conv3d_1/kernel/Regularizer/Square:y:0*conv3d_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!conv3d_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    �
conv3d_1/kernel/Regularizer/mulMul*conv3d_1/kernel/Regularizer/mul/x:output:0(conv3d_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:���������22@�
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp2^conv3d_1/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������22@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp2f
1conv3d_1/kernel/Regularizer/Square/ReadVariableOp1conv3d_1/kernel/Regularizer/Square/ReadVariableOp:[ W
3
_output_shapes!
:���������22@
 
_user_specified_nameinputs
�
o
C__inference_add_loss_layer_call_and_return_conditional_losses_16824

inputs
identity

identity_1=
IdentityIdentityinputs*
T0*
_output_shapes
: ?

Identity_1Identityinputs*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: :> :

_output_shapes
: 
 
_user_specified_nameinputs
�
K
/__inference_max_pooling3d_2_layer_call_fn_17057

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
J__inference_max_pooling3d_2_layer_call_and_return_conditional_losses_13157�
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
(__inference_conv3d_6_layer_call_fn_17071

inputs'
unknown:��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :����������*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *L
fGRE
C__inference_conv3d_6_layer_call_and_return_conditional_losses_13337|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :����������
 
_user_specified_nameinputs
�
�
C__inference_conv3d_1_layer_call_and_return_conditional_losses_13219

inputs<
conv3d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�Conv3D/ReadVariableOp�1conv3d_1/kernel/Regularizer/Square/ReadVariableOp�
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:@@*
dtype0�
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������22@*
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
:���������22@\
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:���������22@�
1conv3d_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:@@*
dtype0�
"conv3d_1/kernel/Regularizer/SquareSquare9conv3d_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0**
_output_shapes
:@@~
!conv3d_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
conv3d_1/kernel/Regularizer/SumSum&conv3d_1/kernel/Regularizer/Square:y:0*conv3d_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!conv3d_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    �
conv3d_1/kernel/Regularizer/mulMul*conv3d_1/kernel/Regularizer/mul/x:output:0(conv3d_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: m
IdentityIdentityRelu:activations:0^NoOp*
T0*3
_output_shapes!
:���������22@�
NoOpNoOp^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp2^conv3d_1/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������22@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp2f
1conv3d_1/kernel/Regularizer/Square/ReadVariableOp1conv3d_1/kernel/Regularizer/Square/ReadVariableOp:[ W
3
_output_shapes!
:���������22@
 
_user_specified_nameinputs
�
�
%__inference_dense_layer_call_fn_17144

inputs
unknown:	�$
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
@__inference_dense_layer_call_and_return_conditional_losses_13391o
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
:����������$: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������$
 
_user_specified_nameinputs
�
�
__inference_loss_fn_0_17171V
8conv3d_kernel_regularizer_square_readvariableop_resource:@
identity��/conv3d/kernel/Regularizer/Square/ReadVariableOp�
/conv3d/kernel/Regularizer/Square/ReadVariableOpReadVariableOp8conv3d_kernel_regularizer_square_readvariableop_resource**
_output_shapes
:@*
dtype0�
 conv3d/kernel/Regularizer/SquareSquare7conv3d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0**
_output_shapes
:@|
conv3d/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
conv3d/kernel/Regularizer/SumSum$conv3d/kernel/Regularizer/Square:y:0(conv3d/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: d
conv3d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    �
conv3d/kernel/Regularizer/mulMul(conv3d/kernel/Regularizer/mul/x:output:0&conv3d/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: _
IdentityIdentity!conv3d/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: x
NoOpNoOp0^conv3d/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2b
/conv3d/kernel/Regularizer/Square/ReadVariableOp/conv3d/kernel/Regularizer/Square/ReadVariableOp
�
�
@__inference_model_layer_call_and_return_conditional_losses_13452

inputs*
conv3d_13197:@
conv3d_13199:@,
conv3d_1_13220:@@
conv3d_1_13222:@-
conv3d_2_13244:@�
conv3d_2_13246:	�.
conv3d_3_13267:��
conv3d_3_13269:	�.
conv3d_4_13291:��
conv3d_4_13293:	�.
conv3d_5_13314:��
conv3d_5_13316:	�.
conv3d_6_13338:��
conv3d_6_13340:	�.
conv3d_7_13361:��
conv3d_7_13363:	�
dense_13392:	�$
dense_13394:
identity��conv3d/StatefulPartitionedCall�/conv3d/kernel/Regularizer/Square/ReadVariableOp� conv3d_1/StatefulPartitionedCall�1conv3d_1/kernel/Regularizer/Square/ReadVariableOp� conv3d_2/StatefulPartitionedCall�1conv3d_2/kernel/Regularizer/Square/ReadVariableOp� conv3d_3/StatefulPartitionedCall�1conv3d_3/kernel/Regularizer/Square/ReadVariableOp� conv3d_4/StatefulPartitionedCall�1conv3d_4/kernel/Regularizer/Square/ReadVariableOp� conv3d_5/StatefulPartitionedCall�1conv3d_5/kernel/Regularizer/Square/ReadVariableOp� conv3d_6/StatefulPartitionedCall�1conv3d_6/kernel/Regularizer/Square/ReadVariableOp� conv3d_7/StatefulPartitionedCall�1conv3d_7/kernel/Regularizer/Square/ReadVariableOp�dense/StatefulPartitionedCall�.dense/kernel/Regularizer/Square/ReadVariableOp�
conv3d/StatefulPartitionedCallStatefulPartitionedCallinputsconv3d_13197conv3d_13199*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������22@*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *J
fERC
A__inference_conv3d_layer_call_and_return_conditional_losses_13196�
 conv3d_1/StatefulPartitionedCallStatefulPartitionedCall'conv3d/StatefulPartitionedCall:output:0conv3d_1_13220conv3d_1_13222*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������22@*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *L
fGRE
C__inference_conv3d_1_layer_call_and_return_conditional_losses_13219�
max_pooling3d/PartitionedCallPartitionedCall)conv3d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������
@* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8� *Q
fLRJ
H__inference_max_pooling3d_layer_call_and_return_conditional_losses_13133�
 conv3d_2/StatefulPartitionedCallStatefulPartitionedCall&max_pooling3d/PartitionedCall:output:0conv3d_2_13244conv3d_2_13246*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :���������
�*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *L
fGRE
C__inference_conv3d_2_layer_call_and_return_conditional_losses_13243�
 conv3d_3/StatefulPartitionedCallStatefulPartitionedCall)conv3d_2/StatefulPartitionedCall:output:0conv3d_3_13267conv3d_3_13269*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :���������
�*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *L
fGRE
C__inference_conv3d_3_layer_call_and_return_conditional_losses_13266�
max_pooling3d_1/PartitionedCallPartitionedCall)conv3d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :����������* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8� *S
fNRL
J__inference_max_pooling3d_1_layer_call_and_return_conditional_losses_13145�
 conv3d_4/StatefulPartitionedCallStatefulPartitionedCall(max_pooling3d_1/PartitionedCall:output:0conv3d_4_13291conv3d_4_13293*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :����������*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *L
fGRE
C__inference_conv3d_4_layer_call_and_return_conditional_losses_13290�
 conv3d_5/StatefulPartitionedCallStatefulPartitionedCall)conv3d_4/StatefulPartitionedCall:output:0conv3d_5_13314conv3d_5_13316*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :����������*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *L
fGRE
C__inference_conv3d_5_layer_call_and_return_conditional_losses_13313�
max_pooling3d_2/PartitionedCallPartitionedCall)conv3d_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :����������* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8� *S
fNRL
J__inference_max_pooling3d_2_layer_call_and_return_conditional_losses_13157�
 conv3d_6/StatefulPartitionedCallStatefulPartitionedCall(max_pooling3d_2/PartitionedCall:output:0conv3d_6_13338conv3d_6_13340*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :����������*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *L
fGRE
C__inference_conv3d_6_layer_call_and_return_conditional_losses_13337�
 conv3d_7/StatefulPartitionedCallStatefulPartitionedCall)conv3d_6/StatefulPartitionedCall:output:0conv3d_7_13361conv3d_7_13363*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :����������*$
_read_only_resource_inputs
*4
config_proto$"

CPU

GPU(2*0J 8� *L
fGRE
C__inference_conv3d_7_layer_call_and_return_conditional_losses_13360�
max_pooling3d_3/PartitionedCallPartitionedCall)conv3d_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :����������* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8� *S
fNRL
J__inference_max_pooling3d_3_layer_call_and_return_conditional_losses_13169�
flatten/PartitionedCallPartitionedCall(max_pooling3d_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������$* 
_read_only_resource_inputs
 *4
config_proto$"

CPU

GPU(2*0J 8� *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_13373�
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_13392dense_13394*
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
@__inference_dense_layer_call_and_return_conditional_losses_13391�
/conv3d/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv3d_13197**
_output_shapes
:@*
dtype0�
 conv3d/kernel/Regularizer/SquareSquare7conv3d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0**
_output_shapes
:@|
conv3d/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
conv3d/kernel/Regularizer/SumSum$conv3d/kernel/Regularizer/Square:y:0(conv3d/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: d
conv3d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    �
conv3d/kernel/Regularizer/mulMul(conv3d/kernel/Regularizer/mul/x:output:0&conv3d/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1conv3d_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv3d_1_13220**
_output_shapes
:@@*
dtype0�
"conv3d_1/kernel/Regularizer/SquareSquare9conv3d_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0**
_output_shapes
:@@~
!conv3d_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
conv3d_1/kernel/Regularizer/SumSum&conv3d_1/kernel/Regularizer/Square:y:0*conv3d_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!conv3d_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    �
conv3d_1/kernel/Regularizer/mulMul*conv3d_1/kernel/Regularizer/mul/x:output:0(conv3d_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1conv3d_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv3d_2_13244*+
_output_shapes
:@�*
dtype0�
"conv3d_2/kernel/Regularizer/SquareSquare9conv3d_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*+
_output_shapes
:@�~
!conv3d_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
conv3d_2/kernel/Regularizer/SumSum&conv3d_2/kernel/Regularizer/Square:y:0*conv3d_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!conv3d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    �
conv3d_2/kernel/Regularizer/mulMul*conv3d_2/kernel/Regularizer/mul/x:output:0(conv3d_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1conv3d_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv3d_3_13267*,
_output_shapes
:��*
dtype0�
"conv3d_3/kernel/Regularizer/SquareSquare9conv3d_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*,
_output_shapes
:��~
!conv3d_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
conv3d_3/kernel/Regularizer/SumSum&conv3d_3/kernel/Regularizer/Square:y:0*conv3d_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!conv3d_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    �
conv3d_3/kernel/Regularizer/mulMul*conv3d_3/kernel/Regularizer/mul/x:output:0(conv3d_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1conv3d_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv3d_4_13291*,
_output_shapes
:��*
dtype0�
"conv3d_4/kernel/Regularizer/SquareSquare9conv3d_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*,
_output_shapes
:��~
!conv3d_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
conv3d_4/kernel/Regularizer/SumSum&conv3d_4/kernel/Regularizer/Square:y:0*conv3d_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!conv3d_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    �
conv3d_4/kernel/Regularizer/mulMul*conv3d_4/kernel/Regularizer/mul/x:output:0(conv3d_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1conv3d_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv3d_5_13314*,
_output_shapes
:��*
dtype0�
"conv3d_5/kernel/Regularizer/SquareSquare9conv3d_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*,
_output_shapes
:��~
!conv3d_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
conv3d_5/kernel/Regularizer/SumSum&conv3d_5/kernel/Regularizer/Square:y:0*conv3d_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!conv3d_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    �
conv3d_5/kernel/Regularizer/mulMul*conv3d_5/kernel/Regularizer/mul/x:output:0(conv3d_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1conv3d_6/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv3d_6_13338*,
_output_shapes
:��*
dtype0�
"conv3d_6/kernel/Regularizer/SquareSquare9conv3d_6/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*,
_output_shapes
:��~
!conv3d_6/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
conv3d_6/kernel/Regularizer/SumSum&conv3d_6/kernel/Regularizer/Square:y:0*conv3d_6/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!conv3d_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    �
conv3d_6/kernel/Regularizer/mulMul*conv3d_6/kernel/Regularizer/mul/x:output:0(conv3d_6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
1conv3d_7/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv3d_7_13361*,
_output_shapes
:��*
dtype0�
"conv3d_7/kernel/Regularizer/SquareSquare9conv3d_7/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*,
_output_shapes
:��~
!conv3d_7/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*)
value B"                �
conv3d_7/kernel/Regularizer/SumSum&conv3d_7/kernel/Regularizer/Square:y:0*conv3d_7/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!conv3d_7/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    �
conv3d_7/kernel/Regularizer/mulMul*conv3d_7/kernel/Regularizer/mul/x:output:0(conv3d_7/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: {
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_13392*
_output_shapes
:	�$*
dtype0�
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	�$o
dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       �
dense/kernel/Regularizer/SumSum#dense/kernel/Regularizer/Square:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: c
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    �
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: u
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^conv3d/StatefulPartitionedCall0^conv3d/kernel/Regularizer/Square/ReadVariableOp!^conv3d_1/StatefulPartitionedCall2^conv3d_1/kernel/Regularizer/Square/ReadVariableOp!^conv3d_2/StatefulPartitionedCall2^conv3d_2/kernel/Regularizer/Square/ReadVariableOp!^conv3d_3/StatefulPartitionedCall2^conv3d_3/kernel/Regularizer/Square/ReadVariableOp!^conv3d_4/StatefulPartitionedCall2^conv3d_4/kernel/Regularizer/Square/ReadVariableOp!^conv3d_5/StatefulPartitionedCall2^conv3d_5/kernel/Regularizer/Square/ReadVariableOp!^conv3d_6/StatefulPartitionedCall2^conv3d_6/kernel/Regularizer/Square/ReadVariableOp!^conv3d_7/StatefulPartitionedCall2^conv3d_7/kernel/Regularizer/Square/ReadVariableOp^dense/StatefulPartitionedCall/^dense/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:���������22: : : : : : : : : : : : : : : : : : 2@
conv3d/StatefulPartitionedCallconv3d/StatefulPartitionedCall2b
/conv3d/kernel/Regularizer/Square/ReadVariableOp/conv3d/kernel/Regularizer/Square/ReadVariableOp2D
 conv3d_1/StatefulPartitionedCall conv3d_1/StatefulPartitionedCall2f
1conv3d_1/kernel/Regularizer/Square/ReadVariableOp1conv3d_1/kernel/Regularizer/Square/ReadVariableOp2D
 conv3d_2/StatefulPartitionedCall conv3d_2/StatefulPartitionedCall2f
1conv3d_2/kernel/Regularizer/Square/ReadVariableOp1conv3d_2/kernel/Regularizer/Square/ReadVariableOp2D
 conv3d_3/StatefulPartitionedCall conv3d_3/StatefulPartitionedCall2f
1conv3d_3/kernel/Regularizer/Square/ReadVariableOp1conv3d_3/kernel/Regularizer/Square/ReadVariableOp2D
 conv3d_4/StatefulPartitionedCall conv3d_4/StatefulPartitionedCall2f
1conv3d_4/kernel/Regularizer/Square/ReadVariableOp1conv3d_4/kernel/Regularizer/Square/ReadVariableOp2D
 conv3d_5/StatefulPartitionedCall conv3d_5/StatefulPartitionedCall2f
1conv3d_5/kernel/Regularizer/Square/ReadVariableOp1conv3d_5/kernel/Regularizer/Square/ReadVariableOp2D
 conv3d_6/StatefulPartitionedCall conv3d_6/StatefulPartitionedCall2f
1conv3d_6/kernel/Regularizer/Square/ReadVariableOp1conv3d_6/kernel/Regularizer/Square/ReadVariableOp2D
 conv3d_7/StatefulPartitionedCall conv3d_7/StatefulPartitionedCall2f
1conv3d_7/kernel/Regularizer/Square/ReadVariableOp1conv3d_7/kernel/Regularizer/Square/ReadVariableOp2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2`
.dense/kernel/Regularizer/Square/ReadVariableOp.dense/kernel/Regularizer/Square/ReadVariableOp:[ W
3
_output_shapes!
:���������22
 
_user_specified_nameinputs"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
G
input_2<
serving_default_input_2:0���������22
G
input_3<
serving_default_input_3:0���������22
;
input_40
serving_default_input_4:0���������
;
input_50
serving_default_input_5:0���������9
model0
StatefulPartitionedCall:0���������;
model_10
StatefulPartitionedCall:1���������tensorflow/serving/predict:��
�

layer-0
layer-1
layer-2
layer-3
layer_with_weights-0
layer-4
layer-5
layer-6
layer-7
	layer-8

layer-9
layer-10
layer-11
layer-12
layer-13
layer-14
layer-15
layer-16
layer-17
layer-18
layer-19
layer-20
layer-21
layer-22
layer-23
layer-24
layer-25
layer-26
layer-27
layer-28
layer-29
layer-30
 layer-31
!layer-32
"layer-33
#layer-34
$layer-35
%layer-36
&layer-37
'layer-38
(layer-39
)layer-40
*layer-41
+layer-42
,layer-43
-layer-44
.layer-45
/layer-46
0layer-47
1layer-48
2layer-49
3layer-50
4layer-51
5layer-52
6layer-53
7layer-54
8layer-55
9layer-56
:layer-57
;layer-58
<layer-59
=layer-60
>layer-61
?layer-62
@layer-63
Alayer-64
Blayer-65
Clayer-66
Dlayer-67
Elayer-68
Flayer-69
Glayer-70
Hlayer-71
Ilayer-72
Jlayer-73
Klayer-74
Llayer-75
M	variables
Ntrainable_variables
Oregularization_losses
P	keras_api
Q__call__
*R&call_and_return_all_conditional_losses
S_default_save_signature
T	optimizer
Uloss
V
signatures"
_tf_keras_network
"
_tf_keras_input_layer
"
_tf_keras_input_layer
6
W_init_input_shape"
_tf_keras_input_layer
6
X_init_input_shape"
_tf_keras_input_layer
�
Ylayer-0
Zlayer_with_weights-0
Zlayer-1
[layer_with_weights-1
[layer-2
\layer-3
]layer_with_weights-2
]layer-4
^layer_with_weights-3
^layer-5
_layer-6
`layer_with_weights-4
`layer-7
alayer_with_weights-5
alayer-8
blayer-9
clayer_with_weights-6
clayer-10
dlayer_with_weights-7
dlayer-11
elayer-12
flayer-13
glayer_with_weights-8
glayer-14
h	variables
itrainable_variables
jregularization_losses
k	keras_api
l__call__
*m&call_and_return_all_conditional_losses"
_tf_keras_network
(
n	keras_api"
_tf_keras_layer
(
o	keras_api"
_tf_keras_layer
(
p	keras_api"
_tf_keras_layer
(
q	keras_api"
_tf_keras_layer
(
r	keras_api"
_tf_keras_layer
(
s	keras_api"
_tf_keras_layer
(
t	keras_api"
_tf_keras_layer
(
u	keras_api"
_tf_keras_layer
(
v	keras_api"
_tf_keras_layer
(
w	keras_api"
_tf_keras_layer
(
x	keras_api"
_tf_keras_layer
(
y	keras_api"
_tf_keras_layer
(
z	keras_api"
_tf_keras_layer
(
{	keras_api"
_tf_keras_layer
(
|	keras_api"
_tf_keras_layer
(
}	keras_api"
_tf_keras_layer
(
~	keras_api"
_tf_keras_layer
(
	keras_api"
_tf_keras_layer
)
�	keras_api"
_tf_keras_layer
)
�	keras_api"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
)
�	keras_api"
_tf_keras_layer
)
�	keras_api"
_tf_keras_layer
)
�	keras_api"
_tf_keras_layer
)
�	keras_api"
_tf_keras_layer
)
�	keras_api"
_tf_keras_layer
)
�	keras_api"
_tf_keras_layer
)
�	keras_api"
_tf_keras_layer
)
�	keras_api"
_tf_keras_layer
)
�	keras_api"
_tf_keras_layer
)
�	keras_api"
_tf_keras_layer
)
�	keras_api"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
)
�	keras_api"
_tf_keras_layer
)
�	keras_api"
_tf_keras_layer
)
�	keras_api"
_tf_keras_layer
)
�	keras_api"
_tf_keras_layer
)
�	keras_api"
_tf_keras_layer
)
�	keras_api"
_tf_keras_layer
)
�	keras_api"
_tf_keras_layer
)
�	keras_api"
_tf_keras_layer
)
�	keras_api"
_tf_keras_layer
)
�	keras_api"
_tf_keras_layer
)
�	keras_api"
_tf_keras_layer
)
�	keras_api"
_tf_keras_layer
)
�	keras_api"
_tf_keras_layer
)
�	keras_api"
_tf_keras_layer
)
�	keras_api"
_tf_keras_layer
)
�	keras_api"
_tf_keras_layer
)
�	keras_api"
_tf_keras_layer
)
�	keras_api"
_tf_keras_layer
)
�	keras_api"
_tf_keras_layer
)
�	keras_api"
_tf_keras_layer
)
�	keras_api"
_tf_keras_layer
)
�	keras_api"
_tf_keras_layer
)
�	keras_api"
_tf_keras_layer
)
�	keras_api"
_tf_keras_layer
)
�	keras_api"
_tf_keras_layer
)
�	keras_api"
_tf_keras_layer
)
�	keras_api"
_tf_keras_layer
)
�	keras_api"
_tf_keras_layer
)
�	keras_api"
_tf_keras_layer
)
�	keras_api"
_tf_keras_layer
)
�	keras_api"
_tf_keras_layer
)
�	keras_api"
_tf_keras_layer
)
�	keras_api"
_tf_keras_layer
)
�	keras_api"
_tf_keras_layer
)
�	keras_api"
_tf_keras_layer
)
�	keras_api"
_tf_keras_layer
)
�	keras_api"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17"
trackable_list_wrapper
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
M	variables
Ntrainable_variables
Oregularization_losses
Q__call__
S_default_save_signature
*R&call_and_return_all_conditional_losses
&R"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_1
�trace_2
�trace_32�
'__inference_model_1_layer_call_fn_14410
'__inference_model_1_layer_call_fn_15670
'__inference_model_1_layer_call_fn_15731
'__inference_model_1_layer_call_fn_14891�
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
B__inference_model_1_layer_call_and_return_conditional_losses_16078
B__inference_model_1_layer_call_and_return_conditional_losses_16425
B__inference_model_1_layer_call_and_return_conditional_losses_15162
B__inference_model_1_layer_call_and_return_conditional_losses_15433�
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
 __inference__wrapped_model_13124input_2input_3input_4input_5"�
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
�
	�iter
�beta_1
�beta_2

�decay
�learning_rate	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�"
	optimizer
 "
trackable_dict_wrapper
-
�serving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
"
_tf_keras_input_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias"
_tf_keras_layer
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17"
trackable_list_wrapper
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17"
trackable_list_wrapper
h
�0
�1
�2
�3
�4
�5
�6
�7
�8"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
h	variables
itrainable_variables
jregularization_losses
l__call__
*m&call_and_return_all_conditional_losses
&m"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_1
�trace_2
�trace_32�
%__inference_model_layer_call_fn_13491
%__inference_model_layer_call_fn_16520
%__inference_model_layer_call_fn_16561
%__inference_model_layer_call_fn_13818�
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
 z�trace_0z�trace_1z�trace_2z�trace_3
�
�trace_0
�trace_1
�trace_2
�trace_32�
@__inference_model_layer_call_and_return_conditional_losses_16687
@__inference_model_layer_call_and_return_conditional_losses_16813
@__inference_model_layer_call_and_return_conditional_losses_13926
@__inference_model_layer_call_and_return_conditional_losses_14034�
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
 z�trace_0z�trace_1z�trace_2z�trace_3
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
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
(__inference_add_loss_layer_call_fn_16819�
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
C__inference_add_loss_layer_call_and_return_conditional_losses_16824�
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
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
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
*__inference_add_metric_layer_call_fn_16833�
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
E__inference_add_metric_layer_call_and_return_conditional_losses_16850�
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
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
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
,__inference_add_metric_1_layer_call_fn_16859�
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
G__inference_add_metric_1_layer_call_and_return_conditional_losses_16876�
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
+:)@2conv3d/kernel
:@2conv3d/bias
-:+@@2conv3d_1/kernel
:@2conv3d_1/bias
.:,@�2conv3d_2/kernel
:�2conv3d_2/bias
/:-��2conv3d_3/kernel
:�2conv3d_3/bias
/:-��2conv3d_4/kernel
:�2conv3d_4/bias
/:-��2conv3d_5/kernel
:�2conv3d_5/bias
/:-��2conv3d_6/kernel
:�2conv3d_6/bias
/:-��2conv3d_7/kernel
:�2conv3d_7/bias
:	�$2dense/kernel
:2
dense/bias
 "
trackable_list_wrapper
�
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
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
 31
!32
"33
#34
$35
%36
&37
'38
(39
)40
*41
+42
,43
-44
.45
/46
047
148
249
350
451
552
653
754
855
956
:57
;58
<59
=60
>61
?62
@63
A64
B65
C66
D67
E68
F69
G70
H71
I72
J73
K74
L75"
trackable_list_wrapper
8
�0
�1
�2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
'__inference_model_1_layer_call_fn_14410input_2input_3input_4input_5"�
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
'__inference_model_1_layer_call_fn_15670inputs/0inputs/1inputs/2inputs/3"�
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
'__inference_model_1_layer_call_fn_15731inputs/0inputs/1inputs/2inputs/3"�
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
'__inference_model_1_layer_call_fn_14891input_2input_3input_4input_5"�
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
B__inference_model_1_layer_call_and_return_conditional_losses_16078inputs/0inputs/1inputs/2inputs/3"�
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
B__inference_model_1_layer_call_and_return_conditional_losses_16425inputs/0inputs/1inputs/2inputs/3"�
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
B__inference_model_1_layer_call_and_return_conditional_losses_15162input_2input_3input_4input_5"�
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
B__inference_model_1_layer_call_and_return_conditional_losses_15433input_2input_3input_4input_5"�
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
#__inference_signature_wrapper_15555input_2input_3input_4input_5"�
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
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
(
�0"
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
&__inference_conv3d_layer_call_fn_16885�
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
A__inference_conv3d_layer_call_and_return_conditional_losses_16902�
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
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
(
�0"
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
(__inference_conv3d_1_layer_call_fn_16911�
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
�trace_02�
C__inference_conv3d_1_layer_call_and_return_conditional_losses_16928�
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
 z�trace_0
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
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
-__inference_max_pooling3d_layer_call_fn_16933�
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
 z�trace_0
�
�trace_02�
H__inference_max_pooling3d_layer_call_and_return_conditional_losses_16938�
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
 z�trace_0
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
(__inference_conv3d_2_layer_call_fn_16947�
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
 z�trace_0
�
�trace_02�
C__inference_conv3d_2_layer_call_and_return_conditional_losses_16964�
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
 z�trace_0
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
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
(__inference_conv3d_3_layer_call_fn_16973�
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
 z�trace_0
�
�trace_02�
C__inference_conv3d_3_layer_call_and_return_conditional_losses_16990�
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
 z�trace_0
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
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
/__inference_max_pooling3d_1_layer_call_fn_16995�
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
 z�trace_0
�
�trace_02�
J__inference_max_pooling3d_1_layer_call_and_return_conditional_losses_17000�
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
 z�trace_0
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
(__inference_conv3d_4_layer_call_fn_17009�
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
 z�trace_0
�
�trace_02�
C__inference_conv3d_4_layer_call_and_return_conditional_losses_17026�
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
 z�trace_0
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
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
(__inference_conv3d_5_layer_call_fn_17035�
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
 z�trace_0
�
�trace_02�
C__inference_conv3d_5_layer_call_and_return_conditional_losses_17052�
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
 z�trace_0
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
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
/__inference_max_pooling3d_2_layer_call_fn_17057�
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
 z�trace_0
�
�trace_02�
J__inference_max_pooling3d_2_layer_call_and_return_conditional_losses_17062�
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
 z�trace_0
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
(__inference_conv3d_6_layer_call_fn_17071�
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
 z�trace_0
�
�trace_02�
C__inference_conv3d_6_layer_call_and_return_conditional_losses_17088�
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
 z�trace_0
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
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
(__inference_conv3d_7_layer_call_fn_17097�
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
 z�trace_0
�
�trace_02�
C__inference_conv3d_7_layer_call_and_return_conditional_losses_17114�
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
 z�trace_0
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
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
/__inference_max_pooling3d_3_layer_call_fn_17119�
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
 z�trace_0
�
�trace_02�
J__inference_max_pooling3d_3_layer_call_and_return_conditional_losses_17124�
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
 z�trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
'__inference_flatten_layer_call_fn_17129�
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
 z�trace_0
�
�trace_02�
B__inference_flatten_layer_call_and_return_conditional_losses_17135�
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
 z�trace_0
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
%__inference_dense_layer_call_fn_17144�
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
 z�trace_0
�
�trace_02�
@__inference_dense_layer_call_and_return_conditional_losses_17160�
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
 z�trace_0
�
�trace_02�
__inference_loss_fn_0_17171�
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
annotations� *� z�trace_0
�
�trace_02�
__inference_loss_fn_1_17182�
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
annotations� *� z�trace_0
�
�trace_02�
__inference_loss_fn_2_17193�
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
annotations� *� z�trace_0
�
�trace_02�
__inference_loss_fn_3_17204�
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
annotations� *� z�trace_0
�
�trace_02�
__inference_loss_fn_4_17215�
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
annotations� *� z�trace_0
�
�trace_02�
__inference_loss_fn_5_17226�
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
annotations� *� z�trace_0
�
�trace_02�
__inference_loss_fn_6_17237�
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
annotations� *� z�trace_0
�
�trace_02�
__inference_loss_fn_7_17248�
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
annotations� *� z�trace_0
�
�trace_02�
__inference_loss_fn_8_17259�
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
annotations� *� z�trace_0
 "
trackable_list_wrapper
�
Y0
Z1
[2
\3
]4
^5
_6
`7
a8
b9
c10
d11
e12
f13
g14"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
%__inference_model_layer_call_fn_13491input_1"�
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
%__inference_model_layer_call_fn_16520inputs"�
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
%__inference_model_layer_call_fn_16561inputs"�
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
%__inference_model_layer_call_fn_13818input_1"�
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
@__inference_model_layer_call_and_return_conditional_losses_16687inputs"�
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
@__inference_model_layer_call_and_return_conditional_losses_16813inputs"�
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
@__inference_model_layer_call_and_return_conditional_losses_13926input_1"�
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
@__inference_model_layer_call_and_return_conditional_losses_14034input_1"�
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
(__inference_add_loss_layer_call_fn_16819inputs"�
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
C__inference_add_loss_layer_call_and_return_conditional_losses_16824inputs"�
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
(
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
0
�	total_mse"
trackable_dict_wrapper
�B�
*__inference_add_metric_layer_call_fn_16833inputs"�
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
E__inference_add_metric_layer_call_and_return_conditional_losses_16850inputs"�
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
(
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
0
�	corr_loss"
trackable_dict_wrapper
�B�
,__inference_add_metric_1_layer_call_fn_16859inputs"�
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
G__inference_add_metric_1_layer_call_and_return_conditional_losses_16876inputs"�
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
R
�	variables
�	keras_api

�total

�count"
_tf_keras_metric
R
�	variables
�	keras_api

�total

�count"
_tf_keras_metric
R
�	variables
�	keras_api

�total

�count"
_tf_keras_metric
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
&__inference_conv3d_layer_call_fn_16885inputs"�
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
A__inference_conv3d_layer_call_and_return_conditional_losses_16902inputs"�
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
(
�0"
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
(__inference_conv3d_1_layer_call_fn_16911inputs"�
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
C__inference_conv3d_1_layer_call_and_return_conditional_losses_16928inputs"�
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
-__inference_max_pooling3d_layer_call_fn_16933inputs"�
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
H__inference_max_pooling3d_layer_call_and_return_conditional_losses_16938inputs"�
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
(
�0"
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
(__inference_conv3d_2_layer_call_fn_16947inputs"�
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
C__inference_conv3d_2_layer_call_and_return_conditional_losses_16964inputs"�
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
(
�0"
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
(__inference_conv3d_3_layer_call_fn_16973inputs"�
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
C__inference_conv3d_3_layer_call_and_return_conditional_losses_16990inputs"�
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
/__inference_max_pooling3d_1_layer_call_fn_16995inputs"�
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
J__inference_max_pooling3d_1_layer_call_and_return_conditional_losses_17000inputs"�
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
(
�0"
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
(__inference_conv3d_4_layer_call_fn_17009inputs"�
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
C__inference_conv3d_4_layer_call_and_return_conditional_losses_17026inputs"�
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
(
�0"
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
(__inference_conv3d_5_layer_call_fn_17035inputs"�
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
C__inference_conv3d_5_layer_call_and_return_conditional_losses_17052inputs"�
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
/__inference_max_pooling3d_2_layer_call_fn_17057inputs"�
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
J__inference_max_pooling3d_2_layer_call_and_return_conditional_losses_17062inputs"�
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
(
�0"
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
(__inference_conv3d_6_layer_call_fn_17071inputs"�
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
C__inference_conv3d_6_layer_call_and_return_conditional_losses_17088inputs"�
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
(
�0"
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
(__inference_conv3d_7_layer_call_fn_17097inputs"�
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
C__inference_conv3d_7_layer_call_and_return_conditional_losses_17114inputs"�
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
/__inference_max_pooling3d_3_layer_call_fn_17119inputs"�
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
J__inference_max_pooling3d_3_layer_call_and_return_conditional_losses_17124inputs"�
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
'__inference_flatten_layer_call_fn_17129inputs"�
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
B__inference_flatten_layer_call_and_return_conditional_losses_17135inputs"�
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
(
�0"
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
%__inference_dense_layer_call_fn_17144inputs"�
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
@__inference_dense_layer_call_and_return_conditional_losses_17160inputs"�
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
__inference_loss_fn_0_17171"�
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
__inference_loss_fn_1_17182"�
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
__inference_loss_fn_2_17193"�
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
__inference_loss_fn_3_17204"�
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
__inference_loss_fn_4_17215"�
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
__inference_loss_fn_5_17226"�
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
__inference_loss_fn_6_17237"�
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
__inference_loss_fn_7_17248"�
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
__inference_loss_fn_8_17259"�
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
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2add_metric/total
:  (2add_metric/count
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2add_metric_1/total
:  (2add_metric_1/count
0:.@2Adam/conv3d/kernel/m
:@2Adam/conv3d/bias/m
2:0@@2Adam/conv3d_1/kernel/m
 :@2Adam/conv3d_1/bias/m
3:1@�2Adam/conv3d_2/kernel/m
!:�2Adam/conv3d_2/bias/m
4:2��2Adam/conv3d_3/kernel/m
!:�2Adam/conv3d_3/bias/m
4:2��2Adam/conv3d_4/kernel/m
!:�2Adam/conv3d_4/bias/m
4:2��2Adam/conv3d_5/kernel/m
!:�2Adam/conv3d_5/bias/m
4:2��2Adam/conv3d_6/kernel/m
!:�2Adam/conv3d_6/bias/m
4:2��2Adam/conv3d_7/kernel/m
!:�2Adam/conv3d_7/bias/m
$:"	�$2Adam/dense/kernel/m
:2Adam/dense/bias/m
0:.@2Adam/conv3d/kernel/v
:@2Adam/conv3d/bias/v
2:0@@2Adam/conv3d_1/kernel/v
 :@2Adam/conv3d_1/bias/v
3:1@�2Adam/conv3d_2/kernel/v
!:�2Adam/conv3d_2/bias/v
4:2��2Adam/conv3d_3/kernel/v
!:�2Adam/conv3d_3/bias/v
4:2��2Adam/conv3d_4/kernel/v
!:�2Adam/conv3d_4/bias/v
4:2��2Adam/conv3d_5/kernel/v
!:�2Adam/conv3d_5/bias/v
4:2��2Adam/conv3d_6/kernel/v
!:�2Adam/conv3d_6/bias/v
4:2��2Adam/conv3d_7/kernel/v
!:�2Adam/conv3d_7/bias/v
$:"	�$2Adam/dense/kernel/v
:2Adam/dense/bias/v
J
Constjtf.TrackableConstant
!J	
Const_1jtf.TrackableConstant
!J	
Const_2jtf.TrackableConstant�
 __inference__wrapped_model_13124�2����������������������������
���
���
-�*
input_2���������22
-�*
input_3���������22
!�
input_4���������
!�
input_5���������
� "[�X
(
model�
model���������
,
model_1!�
model_1����������
C__inference_add_loss_layer_call_and_return_conditional_losses_16824D�
�
�
inputs 
� ""�

�
0 
�
�	
1/0 U
(__inference_add_loss_layer_call_fn_16819)�
�
�
inputs 
� "� �
G__inference_add_metric_1_layer_call_and_return_conditional_losses_16876<���
�
�
inputs 
� "�

�
0 
� _
,__inference_add_metric_1_layer_call_fn_16859/���
�
�
inputs 
� "� �
E__inference_add_metric_layer_call_and_return_conditional_losses_16850<���
�
�
inputs 
� "�

�
0 
� ]
*__inference_add_metric_layer_call_fn_16833/���
�
�
inputs 
� "� �
C__inference_conv3d_1_layer_call_and_return_conditional_losses_16928v��;�8
1�.
,�)
inputs���������22@
� "1�.
'�$
0���������22@
� �
(__inference_conv3d_1_layer_call_fn_16911i��;�8
1�.
,�)
inputs���������22@
� "$�!���������22@�
C__inference_conv3d_2_layer_call_and_return_conditional_losses_16964w��;�8
1�.
,�)
inputs���������
@
� "2�/
(�%
0���������
�
� �
(__inference_conv3d_2_layer_call_fn_16947j��;�8
1�.
,�)
inputs���������
@
� "%�"���������
��
C__inference_conv3d_3_layer_call_and_return_conditional_losses_16990x��<�9
2�/
-�*
inputs���������
�
� "2�/
(�%
0���������
�
� �
(__inference_conv3d_3_layer_call_fn_16973k��<�9
2�/
-�*
inputs���������
�
� "%�"���������
��
C__inference_conv3d_4_layer_call_and_return_conditional_losses_17026x��<�9
2�/
-�*
inputs����������
� "2�/
(�%
0����������
� �
(__inference_conv3d_4_layer_call_fn_17009k��<�9
2�/
-�*
inputs����������
� "%�"�����������
C__inference_conv3d_5_layer_call_and_return_conditional_losses_17052x��<�9
2�/
-�*
inputs����������
� "2�/
(�%
0����������
� �
(__inference_conv3d_5_layer_call_fn_17035k��<�9
2�/
-�*
inputs����������
� "%�"�����������
C__inference_conv3d_6_layer_call_and_return_conditional_losses_17088x��<�9
2�/
-�*
inputs����������
� "2�/
(�%
0����������
� �
(__inference_conv3d_6_layer_call_fn_17071k��<�9
2�/
-�*
inputs����������
� "%�"�����������
C__inference_conv3d_7_layer_call_and_return_conditional_losses_17114x��<�9
2�/
-�*
inputs����������
� "2�/
(�%
0����������
� �
(__inference_conv3d_7_layer_call_fn_17097k��<�9
2�/
-�*
inputs����������
� "%�"�����������
A__inference_conv3d_layer_call_and_return_conditional_losses_16902v��;�8
1�.
,�)
inputs���������22
� "1�.
'�$
0���������22@
� �
&__inference_conv3d_layer_call_fn_16885i��;�8
1�.
,�)
inputs���������22
� "$�!���������22@�
@__inference_dense_layer_call_and_return_conditional_losses_17160_��0�-
&�#
!�
inputs����������$
� "%�"
�
0���������
� {
%__inference_dense_layer_call_fn_17144R��0�-
&�#
!�
inputs����������$
� "�����������
B__inference_flatten_layer_call_and_return_conditional_losses_17135f<�9
2�/
-�*
inputs����������
� "&�#
�
0����������$
� �
'__inference_flatten_layer_call_fn_17129Y<�9
2�/
-�*
inputs����������
� "�����������$;
__inference_loss_fn_0_17171��

� 
� "� ;
__inference_loss_fn_1_17182��

� 
� "� ;
__inference_loss_fn_2_17193��

� 
� "� ;
__inference_loss_fn_3_17204��

� 
� "� ;
__inference_loss_fn_4_17215��

� 
� "� ;
__inference_loss_fn_5_17226��

� 
� "� ;
__inference_loss_fn_6_17237��

� 
� "� ;
__inference_loss_fn_7_17248��

� 
� "� ;
__inference_loss_fn_8_17259��

� 
� "� �
J__inference_max_pooling3d_1_layer_call_and_return_conditional_losses_17000�_�\
U�R
P�M
inputsA���������������������������������������������
� "U�R
K�H
0A���������������������������������������������
� �
/__inference_max_pooling3d_1_layer_call_fn_16995�_�\
U�R
P�M
inputsA���������������������������������������������
� "H�EA����������������������������������������������
J__inference_max_pooling3d_2_layer_call_and_return_conditional_losses_17062�_�\
U�R
P�M
inputsA���������������������������������������������
� "U�R
K�H
0A���������������������������������������������
� �
/__inference_max_pooling3d_2_layer_call_fn_17057�_�\
U�R
P�M
inputsA���������������������������������������������
� "H�EA����������������������������������������������
J__inference_max_pooling3d_3_layer_call_and_return_conditional_losses_17124�_�\
U�R
P�M
inputsA���������������������������������������������
� "U�R
K�H
0A���������������������������������������������
� �
/__inference_max_pooling3d_3_layer_call_fn_17119�_�\
U�R
P�M
inputsA���������������������������������������������
� "H�EA����������������������������������������������
H__inference_max_pooling3d_layer_call_and_return_conditional_losses_16938�_�\
U�R
P�M
inputsA���������������������������������������������
� "U�R
K�H
0A���������������������������������������������
� �
-__inference_max_pooling3d_layer_call_fn_16933�_�\
U�R
P�M
inputsA���������������������������������������������
� "H�EA����������������������������������������������
B__inference_model_1_layer_call_and_return_conditional_losses_15162�2����������������������������
���
���
-�*
input_2���������22
-�*
input_3���������22
!�
input_4���������
!�
input_5���������
p 

 
� "Y�V
A�>
�
0/0���������
�
0/1���������
�
�	
1/0 �
B__inference_model_1_layer_call_and_return_conditional_losses_15433�2����������������������������
���
���
-�*
input_2���������22
-�*
input_3���������22
!�
input_4���������
!�
input_5���������
p

 
� "Y�V
A�>
�
0/0���������
�
0/1���������
�
�	
1/0 �
B__inference_model_1_layer_call_and_return_conditional_losses_16078�2����������������������������
���
���
.�+
inputs/0���������22
.�+
inputs/1���������22
"�
inputs/2���������
"�
inputs/3���������
p 

 
� "Y�V
A�>
�
0/0���������
�
0/1���������
�
�	
1/0 �
B__inference_model_1_layer_call_and_return_conditional_losses_16425�2����������������������������
���
���
.�+
inputs/0���������22
.�+
inputs/1���������22
"�
inputs/2���������
"�
inputs/3���������
p

 
� "Y�V
A�>
�
0/0���������
�
0/1���������
�
�	
1/0 �
'__inference_model_1_layer_call_fn_14410�2����������������������������
���
���
-�*
input_2���������22
-�*
input_3���������22
!�
input_4���������
!�
input_5���������
p 

 
� "=�:
�
0���������
�
1����������
'__inference_model_1_layer_call_fn_14891�2����������������������������
���
���
-�*
input_2���������22
-�*
input_3���������22
!�
input_4���������
!�
input_5���������
p

 
� "=�:
�
0���������
�
1����������
'__inference_model_1_layer_call_fn_15670�2����������������������������
���
���
.�+
inputs/0���������22
.�+
inputs/1���������22
"�
inputs/2���������
"�
inputs/3���������
p 

 
� "=�:
�
0���������
�
1����������
'__inference_model_1_layer_call_fn_15731�2����������������������������
���
���
.�+
inputs/0���������22
.�+
inputs/1���������22
"�
inputs/2���������
"�
inputs/3���������
p

 
� "=�:
�
0���������
�
1����������
@__inference_model_layer_call_and_return_conditional_losses_13926�$������������������D�A
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
@__inference_model_layer_call_and_return_conditional_losses_14034�$������������������D�A
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
@__inference_model_layer_call_and_return_conditional_losses_16687�$������������������C�@
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
@__inference_model_layer_call_and_return_conditional_losses_16813�$������������������C�@
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
%__inference_model_layer_call_fn_13491�$������������������D�A
:�7
-�*
input_1���������22
p 

 
� "�����������
%__inference_model_layer_call_fn_13818�$������������������D�A
:�7
-�*
input_1���������22
p

 
� "�����������
%__inference_model_layer_call_fn_16520�$������������������C�@
9�6
,�)
inputs���������22
p 

 
� "�����������
%__inference_model_layer_call_fn_16561�$������������������C�@
9�6
,�)
inputs���������22
p

 
� "�����������
#__inference_signature_wrapper_15555�2����������������������������
� 
���
8
input_2-�*
input_2���������22
8
input_3-�*
input_3���������22
,
input_4!�
input_4���������
,
input_5!�
input_5���������"[�X
(
model�
model���������
,
model_1!�
model_1���������