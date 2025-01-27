��
��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
�
BiasAdd

value"T	
bias"T
output"T""
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
�
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

�
DepthwiseConv2dNative

input"T
filter"T
output"T"
Ttype:
2"
strides	list(int)",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

$
DisableCopyOnRead
resource�
�
FusedBatchNormV3
x"T

scale"U
offset"U	
mean"U
variance"U
y"T

batch_mean"U
batch_variance"U
reserve_space_1"U
reserve_space_2"U
reserve_space_3"U"
Ttype:
2"
Utype:
2"
epsilonfloat%��8"&
exponential_avg_factorfloat%  �?";
data_formatstringNHWC:
NHWCNCHWNDHWCNCDHW"
is_trainingbool(
.
Identity

input"T
output"T"	
Ttype
u
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
2	
�
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( ""
Ttype:
2	"
Tidxtype0:
2	
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �
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
Relu6
features"T
activations"T"
Ttype:
2	
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
9
Softmax
logits"T
softmax"T"
Ttype:
2
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
 �"serve*2.12.02v2.12.0-rc1-12-g0db597d0d758��
~
Adam/dense_8/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_8/bias/v
w
'Adam/dense_8/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_8/bias/v*
_output_shapes
:*
dtype0
�
Adam/dense_8/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:c*&
shared_nameAdam/dense_8/kernel/v

)Adam/dense_8/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_8/kernel/v*
_output_shapes

:c*
dtype0
�
Adam/CBlock_4_BatchNorm/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:c*/
shared_name Adam/CBlock_4_BatchNorm/beta/v
�
2Adam/CBlock_4_BatchNorm/beta/v/Read/ReadVariableOpReadVariableOpAdam/CBlock_4_BatchNorm/beta/v*
_output_shapes
:c*
dtype0
�
Adam/CBlock_4_BatchNorm/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:c*0
shared_name!Adam/CBlock_4_BatchNorm/gamma/v
�
3Adam/CBlock_4_BatchNorm/gamma/v/Read/ReadVariableOpReadVariableOpAdam/CBlock_4_BatchNorm/gamma/v*
_output_shapes
:c*
dtype0
�
Adam/CBlock_4_SepConv2D/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:c*/
shared_name Adam/CBlock_4_SepConv2D/bias/v
�
2Adam/CBlock_4_SepConv2D/bias/v/Read/ReadVariableOpReadVariableOpAdam/CBlock_4_SepConv2D/bias/v*
_output_shapes
:c*
dtype0
�
*Adam/CBlock_4_SepConv2D/pointwise_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:Jc*;
shared_name,*Adam/CBlock_4_SepConv2D/pointwise_kernel/v
�
>Adam/CBlock_4_SepConv2D/pointwise_kernel/v/Read/ReadVariableOpReadVariableOp*Adam/CBlock_4_SepConv2D/pointwise_kernel/v*&
_output_shapes
:Jc*
dtype0
�
*Adam/CBlock_4_SepConv2D/depthwise_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:J*;
shared_name,*Adam/CBlock_4_SepConv2D/depthwise_kernel/v
�
>Adam/CBlock_4_SepConv2D/depthwise_kernel/v/Read/ReadVariableOpReadVariableOp*Adam/CBlock_4_SepConv2D/depthwise_kernel/v*&
_output_shapes
:J*
dtype0
�
Adam/CBlock_3_BatchNorm/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:J*/
shared_name Adam/CBlock_3_BatchNorm/beta/v
�
2Adam/CBlock_3_BatchNorm/beta/v/Read/ReadVariableOpReadVariableOpAdam/CBlock_3_BatchNorm/beta/v*
_output_shapes
:J*
dtype0
�
Adam/CBlock_3_BatchNorm/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:J*0
shared_name!Adam/CBlock_3_BatchNorm/gamma/v
�
3Adam/CBlock_3_BatchNorm/gamma/v/Read/ReadVariableOpReadVariableOpAdam/CBlock_3_BatchNorm/gamma/v*
_output_shapes
:J*
dtype0
�
Adam/CBlock_3_SepConv2D/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:J*/
shared_name Adam/CBlock_3_SepConv2D/bias/v
�
2Adam/CBlock_3_SepConv2D/bias/v/Read/ReadVariableOpReadVariableOpAdam/CBlock_3_SepConv2D/bias/v*
_output_shapes
:J*
dtype0
�
*Adam/CBlock_3_SepConv2D/pointwise_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:1J*;
shared_name,*Adam/CBlock_3_SepConv2D/pointwise_kernel/v
�
>Adam/CBlock_3_SepConv2D/pointwise_kernel/v/Read/ReadVariableOpReadVariableOp*Adam/CBlock_3_SepConv2D/pointwise_kernel/v*&
_output_shapes
:1J*
dtype0
�
*Adam/CBlock_3_SepConv2D/depthwise_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:1*;
shared_name,*Adam/CBlock_3_SepConv2D/depthwise_kernel/v
�
>Adam/CBlock_3_SepConv2D/depthwise_kernel/v/Read/ReadVariableOpReadVariableOp*Adam/CBlock_3_SepConv2D/depthwise_kernel/v*&
_output_shapes
:1*
dtype0
�
Adam/CBlock_2_BatchNorm/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:1*/
shared_name Adam/CBlock_2_BatchNorm/beta/v
�
2Adam/CBlock_2_BatchNorm/beta/v/Read/ReadVariableOpReadVariableOpAdam/CBlock_2_BatchNorm/beta/v*
_output_shapes
:1*
dtype0
�
Adam/CBlock_2_BatchNorm/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:1*0
shared_name!Adam/CBlock_2_BatchNorm/gamma/v
�
3Adam/CBlock_2_BatchNorm/gamma/v/Read/ReadVariableOpReadVariableOpAdam/CBlock_2_BatchNorm/gamma/v*
_output_shapes
:1*
dtype0
�
Adam/CBlock_2_SepConv2D/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:1*/
shared_name Adam/CBlock_2_SepConv2D/bias/v
�
2Adam/CBlock_2_SepConv2D/bias/v/Read/ReadVariableOpReadVariableOpAdam/CBlock_2_SepConv2D/bias/v*
_output_shapes
:1*
dtype0
�
*Adam/CBlock_2_SepConv2D/pointwise_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:1*;
shared_name,*Adam/CBlock_2_SepConv2D/pointwise_kernel/v
�
>Adam/CBlock_2_SepConv2D/pointwise_kernel/v/Read/ReadVariableOpReadVariableOp*Adam/CBlock_2_SepConv2D/pointwise_kernel/v*&
_output_shapes
:1*
dtype0
�
*Adam/CBlock_2_SepConv2D/depthwise_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*;
shared_name,*Adam/CBlock_2_SepConv2D/depthwise_kernel/v
�
>Adam/CBlock_2_SepConv2D/depthwise_kernel/v/Read/ReadVariableOpReadVariableOp*Adam/CBlock_2_SepConv2D/depthwise_kernel/v*&
_output_shapes
:*
dtype0
�
Adam/CBlock_1_BatchNorm/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name Adam/CBlock_1_BatchNorm/beta/v
�
2Adam/CBlock_1_BatchNorm/beta/v/Read/ReadVariableOpReadVariableOpAdam/CBlock_1_BatchNorm/beta/v*
_output_shapes
:*
dtype0
�
Adam/CBlock_1_BatchNorm/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Adam/CBlock_1_BatchNorm/gamma/v
�
3Adam/CBlock_1_BatchNorm/gamma/v/Read/ReadVariableOpReadVariableOpAdam/CBlock_1_BatchNorm/gamma/v*
_output_shapes
:*
dtype0
�
Adam/CBlock_1_SepConv2D/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name Adam/CBlock_1_SepConv2D/bias/v
�
2Adam/CBlock_1_SepConv2D/bias/v/Read/ReadVariableOpReadVariableOpAdam/CBlock_1_SepConv2D/bias/v*
_output_shapes
:*
dtype0
�
*Adam/CBlock_1_SepConv2D/pointwise_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*;
shared_name,*Adam/CBlock_1_SepConv2D/pointwise_kernel/v
�
>Adam/CBlock_1_SepConv2D/pointwise_kernel/v/Read/ReadVariableOpReadVariableOp*Adam/CBlock_1_SepConv2D/pointwise_kernel/v*&
_output_shapes
:*
dtype0
�
*Adam/CBlock_1_SepConv2D/depthwise_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*;
shared_name,*Adam/CBlock_1_SepConv2D/depthwise_kernel/v
�
>Adam/CBlock_1_SepConv2D/depthwise_kernel/v/Read/ReadVariableOpReadVariableOp*Adam/CBlock_1_SepConv2D/depthwise_kernel/v*&
_output_shapes
:*
dtype0
z
Adam/Conv0/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/Conv0/bias/v
s
%Adam/Conv0/bias/v/Read/ReadVariableOpReadVariableOpAdam/Conv0/bias/v*
_output_shapes
:*
dtype0
�
Adam/Conv0/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/Conv0/kernel/v
�
'Adam/Conv0/kernel/v/Read/ReadVariableOpReadVariableOpAdam/Conv0/kernel/v*&
_output_shapes
:*
dtype0
~
Adam/dense_8/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_8/bias/m
w
'Adam/dense_8/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_8/bias/m*
_output_shapes
:*
dtype0
�
Adam/dense_8/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:c*&
shared_nameAdam/dense_8/kernel/m

)Adam/dense_8/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_8/kernel/m*
_output_shapes

:c*
dtype0
�
Adam/CBlock_4_BatchNorm/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:c*/
shared_name Adam/CBlock_4_BatchNorm/beta/m
�
2Adam/CBlock_4_BatchNorm/beta/m/Read/ReadVariableOpReadVariableOpAdam/CBlock_4_BatchNorm/beta/m*
_output_shapes
:c*
dtype0
�
Adam/CBlock_4_BatchNorm/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:c*0
shared_name!Adam/CBlock_4_BatchNorm/gamma/m
�
3Adam/CBlock_4_BatchNorm/gamma/m/Read/ReadVariableOpReadVariableOpAdam/CBlock_4_BatchNorm/gamma/m*
_output_shapes
:c*
dtype0
�
Adam/CBlock_4_SepConv2D/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:c*/
shared_name Adam/CBlock_4_SepConv2D/bias/m
�
2Adam/CBlock_4_SepConv2D/bias/m/Read/ReadVariableOpReadVariableOpAdam/CBlock_4_SepConv2D/bias/m*
_output_shapes
:c*
dtype0
�
*Adam/CBlock_4_SepConv2D/pointwise_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:Jc*;
shared_name,*Adam/CBlock_4_SepConv2D/pointwise_kernel/m
�
>Adam/CBlock_4_SepConv2D/pointwise_kernel/m/Read/ReadVariableOpReadVariableOp*Adam/CBlock_4_SepConv2D/pointwise_kernel/m*&
_output_shapes
:Jc*
dtype0
�
*Adam/CBlock_4_SepConv2D/depthwise_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:J*;
shared_name,*Adam/CBlock_4_SepConv2D/depthwise_kernel/m
�
>Adam/CBlock_4_SepConv2D/depthwise_kernel/m/Read/ReadVariableOpReadVariableOp*Adam/CBlock_4_SepConv2D/depthwise_kernel/m*&
_output_shapes
:J*
dtype0
�
Adam/CBlock_3_BatchNorm/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:J*/
shared_name Adam/CBlock_3_BatchNorm/beta/m
�
2Adam/CBlock_3_BatchNorm/beta/m/Read/ReadVariableOpReadVariableOpAdam/CBlock_3_BatchNorm/beta/m*
_output_shapes
:J*
dtype0
�
Adam/CBlock_3_BatchNorm/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:J*0
shared_name!Adam/CBlock_3_BatchNorm/gamma/m
�
3Adam/CBlock_3_BatchNorm/gamma/m/Read/ReadVariableOpReadVariableOpAdam/CBlock_3_BatchNorm/gamma/m*
_output_shapes
:J*
dtype0
�
Adam/CBlock_3_SepConv2D/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:J*/
shared_name Adam/CBlock_3_SepConv2D/bias/m
�
2Adam/CBlock_3_SepConv2D/bias/m/Read/ReadVariableOpReadVariableOpAdam/CBlock_3_SepConv2D/bias/m*
_output_shapes
:J*
dtype0
�
*Adam/CBlock_3_SepConv2D/pointwise_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:1J*;
shared_name,*Adam/CBlock_3_SepConv2D/pointwise_kernel/m
�
>Adam/CBlock_3_SepConv2D/pointwise_kernel/m/Read/ReadVariableOpReadVariableOp*Adam/CBlock_3_SepConv2D/pointwise_kernel/m*&
_output_shapes
:1J*
dtype0
�
*Adam/CBlock_3_SepConv2D/depthwise_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:1*;
shared_name,*Adam/CBlock_3_SepConv2D/depthwise_kernel/m
�
>Adam/CBlock_3_SepConv2D/depthwise_kernel/m/Read/ReadVariableOpReadVariableOp*Adam/CBlock_3_SepConv2D/depthwise_kernel/m*&
_output_shapes
:1*
dtype0
�
Adam/CBlock_2_BatchNorm/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:1*/
shared_name Adam/CBlock_2_BatchNorm/beta/m
�
2Adam/CBlock_2_BatchNorm/beta/m/Read/ReadVariableOpReadVariableOpAdam/CBlock_2_BatchNorm/beta/m*
_output_shapes
:1*
dtype0
�
Adam/CBlock_2_BatchNorm/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:1*0
shared_name!Adam/CBlock_2_BatchNorm/gamma/m
�
3Adam/CBlock_2_BatchNorm/gamma/m/Read/ReadVariableOpReadVariableOpAdam/CBlock_2_BatchNorm/gamma/m*
_output_shapes
:1*
dtype0
�
Adam/CBlock_2_SepConv2D/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:1*/
shared_name Adam/CBlock_2_SepConv2D/bias/m
�
2Adam/CBlock_2_SepConv2D/bias/m/Read/ReadVariableOpReadVariableOpAdam/CBlock_2_SepConv2D/bias/m*
_output_shapes
:1*
dtype0
�
*Adam/CBlock_2_SepConv2D/pointwise_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:1*;
shared_name,*Adam/CBlock_2_SepConv2D/pointwise_kernel/m
�
>Adam/CBlock_2_SepConv2D/pointwise_kernel/m/Read/ReadVariableOpReadVariableOp*Adam/CBlock_2_SepConv2D/pointwise_kernel/m*&
_output_shapes
:1*
dtype0
�
*Adam/CBlock_2_SepConv2D/depthwise_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*;
shared_name,*Adam/CBlock_2_SepConv2D/depthwise_kernel/m
�
>Adam/CBlock_2_SepConv2D/depthwise_kernel/m/Read/ReadVariableOpReadVariableOp*Adam/CBlock_2_SepConv2D/depthwise_kernel/m*&
_output_shapes
:*
dtype0
�
Adam/CBlock_1_BatchNorm/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name Adam/CBlock_1_BatchNorm/beta/m
�
2Adam/CBlock_1_BatchNorm/beta/m/Read/ReadVariableOpReadVariableOpAdam/CBlock_1_BatchNorm/beta/m*
_output_shapes
:*
dtype0
�
Adam/CBlock_1_BatchNorm/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Adam/CBlock_1_BatchNorm/gamma/m
�
3Adam/CBlock_1_BatchNorm/gamma/m/Read/ReadVariableOpReadVariableOpAdam/CBlock_1_BatchNorm/gamma/m*
_output_shapes
:*
dtype0
�
Adam/CBlock_1_SepConv2D/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name Adam/CBlock_1_SepConv2D/bias/m
�
2Adam/CBlock_1_SepConv2D/bias/m/Read/ReadVariableOpReadVariableOpAdam/CBlock_1_SepConv2D/bias/m*
_output_shapes
:*
dtype0
�
*Adam/CBlock_1_SepConv2D/pointwise_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*;
shared_name,*Adam/CBlock_1_SepConv2D/pointwise_kernel/m
�
>Adam/CBlock_1_SepConv2D/pointwise_kernel/m/Read/ReadVariableOpReadVariableOp*Adam/CBlock_1_SepConv2D/pointwise_kernel/m*&
_output_shapes
:*
dtype0
�
*Adam/CBlock_1_SepConv2D/depthwise_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*;
shared_name,*Adam/CBlock_1_SepConv2D/depthwise_kernel/m
�
>Adam/CBlock_1_SepConv2D/depthwise_kernel/m/Read/ReadVariableOpReadVariableOp*Adam/CBlock_1_SepConv2D/depthwise_kernel/m*&
_output_shapes
:*
dtype0
z
Adam/Conv0/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/Conv0/bias/m
s
%Adam/Conv0/bias/m/Read/ReadVariableOpReadVariableOpAdam/Conv0/bias/m*
_output_shapes
:*
dtype0
�
Adam/Conv0/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/Conv0/kernel/m
�
'Adam/Conv0/kernel/m/Read/ReadVariableOpReadVariableOpAdam/Conv0/kernel/m*&
_output_shapes
:*
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
p
dense_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_8/bias
i
 dense_8/bias/Read/ReadVariableOpReadVariableOpdense_8/bias*
_output_shapes
:*
dtype0
x
dense_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:c*
shared_namedense_8/kernel
q
"dense_8/kernel/Read/ReadVariableOpReadVariableOpdense_8/kernel*
_output_shapes

:c*
dtype0
�
"CBlock_4_BatchNorm/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:c*3
shared_name$"CBlock_4_BatchNorm/moving_variance
�
6CBlock_4_BatchNorm/moving_variance/Read/ReadVariableOpReadVariableOp"CBlock_4_BatchNorm/moving_variance*
_output_shapes
:c*
dtype0
�
CBlock_4_BatchNorm/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:c*/
shared_name CBlock_4_BatchNorm/moving_mean
�
2CBlock_4_BatchNorm/moving_mean/Read/ReadVariableOpReadVariableOpCBlock_4_BatchNorm/moving_mean*
_output_shapes
:c*
dtype0
�
CBlock_4_BatchNorm/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:c*(
shared_nameCBlock_4_BatchNorm/beta

+CBlock_4_BatchNorm/beta/Read/ReadVariableOpReadVariableOpCBlock_4_BatchNorm/beta*
_output_shapes
:c*
dtype0
�
CBlock_4_BatchNorm/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:c*)
shared_nameCBlock_4_BatchNorm/gamma
�
,CBlock_4_BatchNorm/gamma/Read/ReadVariableOpReadVariableOpCBlock_4_BatchNorm/gamma*
_output_shapes
:c*
dtype0
�
CBlock_4_SepConv2D/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:c*(
shared_nameCBlock_4_SepConv2D/bias

+CBlock_4_SepConv2D/bias/Read/ReadVariableOpReadVariableOpCBlock_4_SepConv2D/bias*
_output_shapes
:c*
dtype0
�
#CBlock_4_SepConv2D/pointwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:Jc*4
shared_name%#CBlock_4_SepConv2D/pointwise_kernel
�
7CBlock_4_SepConv2D/pointwise_kernel/Read/ReadVariableOpReadVariableOp#CBlock_4_SepConv2D/pointwise_kernel*&
_output_shapes
:Jc*
dtype0
�
#CBlock_4_SepConv2D/depthwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:J*4
shared_name%#CBlock_4_SepConv2D/depthwise_kernel
�
7CBlock_4_SepConv2D/depthwise_kernel/Read/ReadVariableOpReadVariableOp#CBlock_4_SepConv2D/depthwise_kernel*&
_output_shapes
:J*
dtype0
�
"CBlock_3_BatchNorm/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:J*3
shared_name$"CBlock_3_BatchNorm/moving_variance
�
6CBlock_3_BatchNorm/moving_variance/Read/ReadVariableOpReadVariableOp"CBlock_3_BatchNorm/moving_variance*
_output_shapes
:J*
dtype0
�
CBlock_3_BatchNorm/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:J*/
shared_name CBlock_3_BatchNorm/moving_mean
�
2CBlock_3_BatchNorm/moving_mean/Read/ReadVariableOpReadVariableOpCBlock_3_BatchNorm/moving_mean*
_output_shapes
:J*
dtype0
�
CBlock_3_BatchNorm/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:J*(
shared_nameCBlock_3_BatchNorm/beta

+CBlock_3_BatchNorm/beta/Read/ReadVariableOpReadVariableOpCBlock_3_BatchNorm/beta*
_output_shapes
:J*
dtype0
�
CBlock_3_BatchNorm/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:J*)
shared_nameCBlock_3_BatchNorm/gamma
�
,CBlock_3_BatchNorm/gamma/Read/ReadVariableOpReadVariableOpCBlock_3_BatchNorm/gamma*
_output_shapes
:J*
dtype0
�
CBlock_3_SepConv2D/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:J*(
shared_nameCBlock_3_SepConv2D/bias

+CBlock_3_SepConv2D/bias/Read/ReadVariableOpReadVariableOpCBlock_3_SepConv2D/bias*
_output_shapes
:J*
dtype0
�
#CBlock_3_SepConv2D/pointwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:1J*4
shared_name%#CBlock_3_SepConv2D/pointwise_kernel
�
7CBlock_3_SepConv2D/pointwise_kernel/Read/ReadVariableOpReadVariableOp#CBlock_3_SepConv2D/pointwise_kernel*&
_output_shapes
:1J*
dtype0
�
#CBlock_3_SepConv2D/depthwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:1*4
shared_name%#CBlock_3_SepConv2D/depthwise_kernel
�
7CBlock_3_SepConv2D/depthwise_kernel/Read/ReadVariableOpReadVariableOp#CBlock_3_SepConv2D/depthwise_kernel*&
_output_shapes
:1*
dtype0
�
"CBlock_2_BatchNorm/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:1*3
shared_name$"CBlock_2_BatchNorm/moving_variance
�
6CBlock_2_BatchNorm/moving_variance/Read/ReadVariableOpReadVariableOp"CBlock_2_BatchNorm/moving_variance*
_output_shapes
:1*
dtype0
�
CBlock_2_BatchNorm/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:1*/
shared_name CBlock_2_BatchNorm/moving_mean
�
2CBlock_2_BatchNorm/moving_mean/Read/ReadVariableOpReadVariableOpCBlock_2_BatchNorm/moving_mean*
_output_shapes
:1*
dtype0
�
CBlock_2_BatchNorm/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:1*(
shared_nameCBlock_2_BatchNorm/beta

+CBlock_2_BatchNorm/beta/Read/ReadVariableOpReadVariableOpCBlock_2_BatchNorm/beta*
_output_shapes
:1*
dtype0
�
CBlock_2_BatchNorm/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:1*)
shared_nameCBlock_2_BatchNorm/gamma
�
,CBlock_2_BatchNorm/gamma/Read/ReadVariableOpReadVariableOpCBlock_2_BatchNorm/gamma*
_output_shapes
:1*
dtype0
�
CBlock_2_SepConv2D/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:1*(
shared_nameCBlock_2_SepConv2D/bias

+CBlock_2_SepConv2D/bias/Read/ReadVariableOpReadVariableOpCBlock_2_SepConv2D/bias*
_output_shapes
:1*
dtype0
�
#CBlock_2_SepConv2D/pointwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:1*4
shared_name%#CBlock_2_SepConv2D/pointwise_kernel
�
7CBlock_2_SepConv2D/pointwise_kernel/Read/ReadVariableOpReadVariableOp#CBlock_2_SepConv2D/pointwise_kernel*&
_output_shapes
:1*
dtype0
�
#CBlock_2_SepConv2D/depthwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#CBlock_2_SepConv2D/depthwise_kernel
�
7CBlock_2_SepConv2D/depthwise_kernel/Read/ReadVariableOpReadVariableOp#CBlock_2_SepConv2D/depthwise_kernel*&
_output_shapes
:*
dtype0
�
"CBlock_1_BatchNorm/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"CBlock_1_BatchNorm/moving_variance
�
6CBlock_1_BatchNorm/moving_variance/Read/ReadVariableOpReadVariableOp"CBlock_1_BatchNorm/moving_variance*
_output_shapes
:*
dtype0
�
CBlock_1_BatchNorm/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name CBlock_1_BatchNorm/moving_mean
�
2CBlock_1_BatchNorm/moving_mean/Read/ReadVariableOpReadVariableOpCBlock_1_BatchNorm/moving_mean*
_output_shapes
:*
dtype0
�
CBlock_1_BatchNorm/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameCBlock_1_BatchNorm/beta

+CBlock_1_BatchNorm/beta/Read/ReadVariableOpReadVariableOpCBlock_1_BatchNorm/beta*
_output_shapes
:*
dtype0
�
CBlock_1_BatchNorm/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameCBlock_1_BatchNorm/gamma
�
,CBlock_1_BatchNorm/gamma/Read/ReadVariableOpReadVariableOpCBlock_1_BatchNorm/gamma*
_output_shapes
:*
dtype0
�
CBlock_1_SepConv2D/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameCBlock_1_SepConv2D/bias

+CBlock_1_SepConv2D/bias/Read/ReadVariableOpReadVariableOpCBlock_1_SepConv2D/bias*
_output_shapes
:*
dtype0
�
#CBlock_1_SepConv2D/pointwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#CBlock_1_SepConv2D/pointwise_kernel
�
7CBlock_1_SepConv2D/pointwise_kernel/Read/ReadVariableOpReadVariableOp#CBlock_1_SepConv2D/pointwise_kernel*&
_output_shapes
:*
dtype0
�
#CBlock_1_SepConv2D/depthwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#CBlock_1_SepConv2D/depthwise_kernel
�
7CBlock_1_SepConv2D/depthwise_kernel/Read/ReadVariableOpReadVariableOp#CBlock_1_SepConv2D/depthwise_kernel*&
_output_shapes
:*
dtype0
l

Conv0/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
Conv0/bias
e
Conv0/bias/Read/ReadVariableOpReadVariableOp
Conv0/bias*
_output_shapes
:*
dtype0
|
Conv0/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameConv0/kernel
u
 Conv0/kernel/Read/ReadVariableOpReadVariableOpConv0/kernel*&
_output_shapes
:*
dtype0
�
serving_default_input_layerPlaceholder*/
_output_shapes
:���������HX*
dtype0*$
shape:���������HX
�

StatefulPartitionedCallStatefulPartitionedCallserving_default_input_layerConv0/kernel
Conv0/bias#CBlock_1_SepConv2D/depthwise_kernel#CBlock_1_SepConv2D/pointwise_kernelCBlock_1_SepConv2D/biasCBlock_1_BatchNorm/gammaCBlock_1_BatchNorm/betaCBlock_1_BatchNorm/moving_mean"CBlock_1_BatchNorm/moving_variance#CBlock_2_SepConv2D/depthwise_kernel#CBlock_2_SepConv2D/pointwise_kernelCBlock_2_SepConv2D/biasCBlock_2_BatchNorm/gammaCBlock_2_BatchNorm/betaCBlock_2_BatchNorm/moving_mean"CBlock_2_BatchNorm/moving_variance#CBlock_3_SepConv2D/depthwise_kernel#CBlock_3_SepConv2D/pointwise_kernelCBlock_3_SepConv2D/biasCBlock_3_BatchNorm/gammaCBlock_3_BatchNorm/betaCBlock_3_BatchNorm/moving_mean"CBlock_3_BatchNorm/moving_variance#CBlock_4_SepConv2D/depthwise_kernel#CBlock_4_SepConv2D/pointwise_kernelCBlock_4_SepConv2D/biasCBlock_4_BatchNorm/gammaCBlock_4_BatchNorm/betaCBlock_4_BatchNorm/moving_mean"CBlock_4_BatchNorm/moving_variancedense_8/kerneldense_8/bias*,
Tin%
#2!*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*B
_read_only_resource_inputs$
" 	
 *0
config_proto 

CPU

GPU2*0J 8� *.
f)R'
%__inference_signature_wrapper_1689886

NoOpNoOp
Ҵ
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*��
value��B�� B��
�
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
layer-6
layer_with_weights-5
layer-7
	layer_with_weights-6
	layer-8

layer-9
layer_with_weights-7
layer-10
layer_with_weights-8
layer-11
layer-12
layer-13
layer_with_weights-9
layer-14
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures*
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
 bias
 !_jit_compiled_convolution_op*
�
"	variables
#trainable_variables
$regularization_losses
%	keras_api
&__call__
*'&call_and_return_all_conditional_losses
(depthwise_kernel
)pointwise_kernel
*bias
 +_jit_compiled_convolution_op*
�
,	variables
-trainable_variables
.regularization_losses
/	keras_api
0__call__
*1&call_and_return_all_conditional_losses
2axis
	3gamma
4beta
5moving_mean
6moving_variance*
�
7	variables
8trainable_variables
9regularization_losses
:	keras_api
;__call__
*<&call_and_return_all_conditional_losses* 
�
=	variables
>trainable_variables
?regularization_losses
@	keras_api
A__call__
*B&call_and_return_all_conditional_losses
Cdepthwise_kernel
Dpointwise_kernel
Ebias
 F_jit_compiled_convolution_op*
�
G	variables
Htrainable_variables
Iregularization_losses
J	keras_api
K__call__
*L&call_and_return_all_conditional_losses
Maxis
	Ngamma
Obeta
Pmoving_mean
Qmoving_variance*
�
R	variables
Strainable_variables
Tregularization_losses
U	keras_api
V__call__
*W&call_and_return_all_conditional_losses* 
�
X	variables
Ytrainable_variables
Zregularization_losses
[	keras_api
\__call__
*]&call_and_return_all_conditional_losses
^depthwise_kernel
_pointwise_kernel
`bias
 a_jit_compiled_convolution_op*
�
b	variables
ctrainable_variables
dregularization_losses
e	keras_api
f__call__
*g&call_and_return_all_conditional_losses
haxis
	igamma
jbeta
kmoving_mean
lmoving_variance*
�
m	variables
ntrainable_variables
oregularization_losses
p	keras_api
q__call__
*r&call_and_return_all_conditional_losses* 
�
s	variables
ttrainable_variables
uregularization_losses
v	keras_api
w__call__
*x&call_and_return_all_conditional_losses
ydepthwise_kernel
zpointwise_kernel
{bias
 |_jit_compiled_convolution_op*
�
}	variables
~trainable_variables
regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis

�gamma
	�beta
�moving_mean
�moving_variance*
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
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias*
�
0
 1
(2
)3
*4
35
46
57
68
C9
D10
E11
N12
O13
P14
Q15
^16
_17
`18
i19
j20
k21
l22
y23
z24
{25
�26
�27
�28
�29
�30
�31*
�
0
 1
(2
)3
*4
35
46
C7
D8
E9
N10
O11
^12
_13
`14
i15
j16
y17
z18
{19
�20
�21
�22
�23*
* 
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
�
	�iter
�beta_1
�beta_2

�decay
�learning_ratem� m�(m�)m�*m�3m�4m�Cm�Dm�Em�Nm�Om�^m�_m�`m�im�jm�ym�zm�{m�	�m�	�m�	�m�	�m�v� v�(v�)v�*v�3v�4v�Cv�Dv�Ev�Nv�Ov�^v�_v�`v�iv�jv�yv�zv�{v�	�v�	�v�	�v�	�v�*

�serving_default* 

0
 1*

0
 1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
\V
VARIABLE_VALUEConv0/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUE
Conv0/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

(0
)1
*2*

(0
)1
*2*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
"	variables
#trainable_variables
$regularization_losses
&__call__
*'&call_and_return_all_conditional_losses
&'"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
}w
VARIABLE_VALUE#CBlock_1_SepConv2D/depthwise_kernel@layer_with_weights-1/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE#CBlock_1_SepConv2D/pointwise_kernel@layer_with_weights-1/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUECBlock_1_SepConv2D/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
 
30
41
52
63*

30
41*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
,	variables
-trainable_variables
.regularization_losses
0__call__
*1&call_and_return_all_conditional_losses
&1"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
ga
VARIABLE_VALUECBlock_1_BatchNorm/gamma5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUECBlock_1_BatchNorm/beta4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUECBlock_1_BatchNorm/moving_mean;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUE"CBlock_1_BatchNorm/moving_variance?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
7	variables
8trainable_variables
9regularization_losses
;__call__
*<&call_and_return_all_conditional_losses
&<"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

C0
D1
E2*

C0
D1
E2*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
=	variables
>trainable_variables
?regularization_losses
A__call__
*B&call_and_return_all_conditional_losses
&B"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
}w
VARIABLE_VALUE#CBlock_2_SepConv2D/depthwise_kernel@layer_with_weights-3/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE#CBlock_2_SepConv2D/pointwise_kernel@layer_with_weights-3/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUECBlock_2_SepConv2D/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
 
N0
O1
P2
Q3*

N0
O1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
G	variables
Htrainable_variables
Iregularization_losses
K__call__
*L&call_and_return_all_conditional_losses
&L"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
ga
VARIABLE_VALUECBlock_2_BatchNorm/gamma5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUECBlock_2_BatchNorm/beta4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUECBlock_2_BatchNorm/moving_mean;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUE"CBlock_2_BatchNorm/moving_variance?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
R	variables
Strainable_variables
Tregularization_losses
V__call__
*W&call_and_return_all_conditional_losses
&W"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

^0
_1
`2*

^0
_1
`2*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
X	variables
Ytrainable_variables
Zregularization_losses
\__call__
*]&call_and_return_all_conditional_losses
&]"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
}w
VARIABLE_VALUE#CBlock_3_SepConv2D/depthwise_kernel@layer_with_weights-5/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE#CBlock_3_SepConv2D/pointwise_kernel@layer_with_weights-5/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUECBlock_3_SepConv2D/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
 
i0
j1
k2
l3*

i0
j1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
b	variables
ctrainable_variables
dregularization_losses
f__call__
*g&call_and_return_all_conditional_losses
&g"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
ga
VARIABLE_VALUECBlock_3_BatchNorm/gamma5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUECBlock_3_BatchNorm/beta4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUECBlock_3_BatchNorm/moving_mean;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUE"CBlock_3_BatchNorm/moving_variance?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
m	variables
ntrainable_variables
oregularization_losses
q__call__
*r&call_and_return_all_conditional_losses
&r"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

y0
z1
{2*

y0
z1
{2*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
s	variables
ttrainable_variables
uregularization_losses
w__call__
*x&call_and_return_all_conditional_losses
&x"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
}w
VARIABLE_VALUE#CBlock_4_SepConv2D/depthwise_kernel@layer_with_weights-7/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE#CBlock_4_SepConv2D/pointwise_kernel@layer_with_weights-7/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUECBlock_4_SepConv2D/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
$
�0
�1
�2
�3*
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
}	variables
~trainable_variables
regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
ga
VARIABLE_VALUECBlock_4_BatchNorm/gamma5layer_with_weights-8/gamma/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUECBlock_4_BatchNorm/beta4layer_with_weights-8/beta/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUECBlock_4_BatchNorm/moving_mean;layer_with_weights-8/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUE"CBlock_4_BatchNorm/moving_variance?layer_with_weights-8/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
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
^X
VARIABLE_VALUEdense_8/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_8/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE*
>
50
61
P2
Q3
k4
l5
�6
�7*
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

50
61*
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

P0
Q1*
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

k0
l1*
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

�0
�1*
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
VARIABLE_VALUEAdam/Conv0/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/Conv0/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE*Adam/CBlock_1_SepConv2D/depthwise_kernel/m\layer_with_weights-1/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE*Adam/CBlock_1_SepConv2D/pointwise_kernel/m\layer_with_weights-1/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEAdam/CBlock_1_SepConv2D/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEAdam/CBlock_1_BatchNorm/gamma/mQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEAdam/CBlock_1_BatchNorm/beta/mPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE*Adam/CBlock_2_SepConv2D/depthwise_kernel/m\layer_with_weights-3/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE*Adam/CBlock_2_SepConv2D/pointwise_kernel/m\layer_with_weights-3/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEAdam/CBlock_2_SepConv2D/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEAdam/CBlock_2_BatchNorm/gamma/mQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEAdam/CBlock_2_BatchNorm/beta/mPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE*Adam/CBlock_3_SepConv2D/depthwise_kernel/m\layer_with_weights-5/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE*Adam/CBlock_3_SepConv2D/pointwise_kernel/m\layer_with_weights-5/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEAdam/CBlock_3_SepConv2D/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEAdam/CBlock_3_BatchNorm/gamma/mQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEAdam/CBlock_3_BatchNorm/beta/mPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE*Adam/CBlock_4_SepConv2D/depthwise_kernel/m\layer_with_weights-7/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE*Adam/CBlock_4_SepConv2D/pointwise_kernel/m\layer_with_weights-7/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEAdam/CBlock_4_SepConv2D/bias/mPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEAdam/CBlock_4_BatchNorm/gamma/mQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEAdam/CBlock_4_BatchNorm/beta/mPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�{
VARIABLE_VALUEAdam/dense_8/kernel/mRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_8/bias/mPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/Conv0/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/Conv0/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE*Adam/CBlock_1_SepConv2D/depthwise_kernel/v\layer_with_weights-1/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE*Adam/CBlock_1_SepConv2D/pointwise_kernel/v\layer_with_weights-1/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEAdam/CBlock_1_SepConv2D/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEAdam/CBlock_1_BatchNorm/gamma/vQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEAdam/CBlock_1_BatchNorm/beta/vPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE*Adam/CBlock_2_SepConv2D/depthwise_kernel/v\layer_with_weights-3/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE*Adam/CBlock_2_SepConv2D/pointwise_kernel/v\layer_with_weights-3/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEAdam/CBlock_2_SepConv2D/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEAdam/CBlock_2_BatchNorm/gamma/vQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEAdam/CBlock_2_BatchNorm/beta/vPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE*Adam/CBlock_3_SepConv2D/depthwise_kernel/v\layer_with_weights-5/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE*Adam/CBlock_3_SepConv2D/pointwise_kernel/v\layer_with_weights-5/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEAdam/CBlock_3_SepConv2D/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEAdam/CBlock_3_BatchNorm/gamma/vQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEAdam/CBlock_3_BatchNorm/beta/vPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE*Adam/CBlock_4_SepConv2D/depthwise_kernel/v\layer_with_weights-7/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE*Adam/CBlock_4_SepConv2D/pointwise_kernel/v\layer_with_weights-7/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEAdam/CBlock_4_SepConv2D/bias/vPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEAdam/CBlock_4_BatchNorm/gamma/vQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEAdam/CBlock_4_BatchNorm/beta/vPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�{
VARIABLE_VALUEAdam/dense_8/kernel/vRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_8/bias/vPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameConv0/kernel
Conv0/bias#CBlock_1_SepConv2D/depthwise_kernel#CBlock_1_SepConv2D/pointwise_kernelCBlock_1_SepConv2D/biasCBlock_1_BatchNorm/gammaCBlock_1_BatchNorm/betaCBlock_1_BatchNorm/moving_mean"CBlock_1_BatchNorm/moving_variance#CBlock_2_SepConv2D/depthwise_kernel#CBlock_2_SepConv2D/pointwise_kernelCBlock_2_SepConv2D/biasCBlock_2_BatchNorm/gammaCBlock_2_BatchNorm/betaCBlock_2_BatchNorm/moving_mean"CBlock_2_BatchNorm/moving_variance#CBlock_3_SepConv2D/depthwise_kernel#CBlock_3_SepConv2D/pointwise_kernelCBlock_3_SepConv2D/biasCBlock_3_BatchNorm/gammaCBlock_3_BatchNorm/betaCBlock_3_BatchNorm/moving_mean"CBlock_3_BatchNorm/moving_variance#CBlock_4_SepConv2D/depthwise_kernel#CBlock_4_SepConv2D/pointwise_kernelCBlock_4_SepConv2D/biasCBlock_4_BatchNorm/gammaCBlock_4_BatchNorm/betaCBlock_4_BatchNorm/moving_mean"CBlock_4_BatchNorm/moving_variancedense_8/kerneldense_8/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotal_1count_1totalcountAdam/Conv0/kernel/mAdam/Conv0/bias/m*Adam/CBlock_1_SepConv2D/depthwise_kernel/m*Adam/CBlock_1_SepConv2D/pointwise_kernel/mAdam/CBlock_1_SepConv2D/bias/mAdam/CBlock_1_BatchNorm/gamma/mAdam/CBlock_1_BatchNorm/beta/m*Adam/CBlock_2_SepConv2D/depthwise_kernel/m*Adam/CBlock_2_SepConv2D/pointwise_kernel/mAdam/CBlock_2_SepConv2D/bias/mAdam/CBlock_2_BatchNorm/gamma/mAdam/CBlock_2_BatchNorm/beta/m*Adam/CBlock_3_SepConv2D/depthwise_kernel/m*Adam/CBlock_3_SepConv2D/pointwise_kernel/mAdam/CBlock_3_SepConv2D/bias/mAdam/CBlock_3_BatchNorm/gamma/mAdam/CBlock_3_BatchNorm/beta/m*Adam/CBlock_4_SepConv2D/depthwise_kernel/m*Adam/CBlock_4_SepConv2D/pointwise_kernel/mAdam/CBlock_4_SepConv2D/bias/mAdam/CBlock_4_BatchNorm/gamma/mAdam/CBlock_4_BatchNorm/beta/mAdam/dense_8/kernel/mAdam/dense_8/bias/mAdam/Conv0/kernel/vAdam/Conv0/bias/v*Adam/CBlock_1_SepConv2D/depthwise_kernel/v*Adam/CBlock_1_SepConv2D/pointwise_kernel/vAdam/CBlock_1_SepConv2D/bias/vAdam/CBlock_1_BatchNorm/gamma/vAdam/CBlock_1_BatchNorm/beta/v*Adam/CBlock_2_SepConv2D/depthwise_kernel/v*Adam/CBlock_2_SepConv2D/pointwise_kernel/vAdam/CBlock_2_SepConv2D/bias/vAdam/CBlock_2_BatchNorm/gamma/vAdam/CBlock_2_BatchNorm/beta/v*Adam/CBlock_3_SepConv2D/depthwise_kernel/v*Adam/CBlock_3_SepConv2D/pointwise_kernel/vAdam/CBlock_3_SepConv2D/bias/vAdam/CBlock_3_BatchNorm/gamma/vAdam/CBlock_3_BatchNorm/beta/v*Adam/CBlock_4_SepConv2D/depthwise_kernel/v*Adam/CBlock_4_SepConv2D/pointwise_kernel/vAdam/CBlock_4_SepConv2D/bias/vAdam/CBlock_4_BatchNorm/gamma/vAdam/CBlock_4_BatchNorm/beta/vAdam/dense_8/kernel/vAdam/dense_8/bias/vConst*f
Tin_
]2[*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *)
f$R"
 __inference__traced_save_1691269
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameConv0/kernel
Conv0/bias#CBlock_1_SepConv2D/depthwise_kernel#CBlock_1_SepConv2D/pointwise_kernelCBlock_1_SepConv2D/biasCBlock_1_BatchNorm/gammaCBlock_1_BatchNorm/betaCBlock_1_BatchNorm/moving_mean"CBlock_1_BatchNorm/moving_variance#CBlock_2_SepConv2D/depthwise_kernel#CBlock_2_SepConv2D/pointwise_kernelCBlock_2_SepConv2D/biasCBlock_2_BatchNorm/gammaCBlock_2_BatchNorm/betaCBlock_2_BatchNorm/moving_mean"CBlock_2_BatchNorm/moving_variance#CBlock_3_SepConv2D/depthwise_kernel#CBlock_3_SepConv2D/pointwise_kernelCBlock_3_SepConv2D/biasCBlock_3_BatchNorm/gammaCBlock_3_BatchNorm/betaCBlock_3_BatchNorm/moving_mean"CBlock_3_BatchNorm/moving_variance#CBlock_4_SepConv2D/depthwise_kernel#CBlock_4_SepConv2D/pointwise_kernelCBlock_4_SepConv2D/biasCBlock_4_BatchNorm/gammaCBlock_4_BatchNorm/betaCBlock_4_BatchNorm/moving_mean"CBlock_4_BatchNorm/moving_variancedense_8/kerneldense_8/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotal_1count_1totalcountAdam/Conv0/kernel/mAdam/Conv0/bias/m*Adam/CBlock_1_SepConv2D/depthwise_kernel/m*Adam/CBlock_1_SepConv2D/pointwise_kernel/mAdam/CBlock_1_SepConv2D/bias/mAdam/CBlock_1_BatchNorm/gamma/mAdam/CBlock_1_BatchNorm/beta/m*Adam/CBlock_2_SepConv2D/depthwise_kernel/m*Adam/CBlock_2_SepConv2D/pointwise_kernel/mAdam/CBlock_2_SepConv2D/bias/mAdam/CBlock_2_BatchNorm/gamma/mAdam/CBlock_2_BatchNorm/beta/m*Adam/CBlock_3_SepConv2D/depthwise_kernel/m*Adam/CBlock_3_SepConv2D/pointwise_kernel/mAdam/CBlock_3_SepConv2D/bias/mAdam/CBlock_3_BatchNorm/gamma/mAdam/CBlock_3_BatchNorm/beta/m*Adam/CBlock_4_SepConv2D/depthwise_kernel/m*Adam/CBlock_4_SepConv2D/pointwise_kernel/mAdam/CBlock_4_SepConv2D/bias/mAdam/CBlock_4_BatchNorm/gamma/mAdam/CBlock_4_BatchNorm/beta/mAdam/dense_8/kernel/mAdam/dense_8/bias/mAdam/Conv0/kernel/vAdam/Conv0/bias/v*Adam/CBlock_1_SepConv2D/depthwise_kernel/v*Adam/CBlock_1_SepConv2D/pointwise_kernel/vAdam/CBlock_1_SepConv2D/bias/vAdam/CBlock_1_BatchNorm/gamma/vAdam/CBlock_1_BatchNorm/beta/v*Adam/CBlock_2_SepConv2D/depthwise_kernel/v*Adam/CBlock_2_SepConv2D/pointwise_kernel/vAdam/CBlock_2_SepConv2D/bias/vAdam/CBlock_2_BatchNorm/gamma/vAdam/CBlock_2_BatchNorm/beta/v*Adam/CBlock_3_SepConv2D/depthwise_kernel/v*Adam/CBlock_3_SepConv2D/pointwise_kernel/vAdam/CBlock_3_SepConv2D/bias/vAdam/CBlock_3_BatchNorm/gamma/vAdam/CBlock_3_BatchNorm/beta/v*Adam/CBlock_4_SepConv2D/depthwise_kernel/v*Adam/CBlock_4_SepConv2D/pointwise_kernel/vAdam/CBlock_4_SepConv2D/bias/vAdam/CBlock_4_BatchNorm/gamma/vAdam/CBlock_4_BatchNorm/beta/vAdam/dense_8/kernel/vAdam/dense_8/bias/v*e
Tin^
\2Z*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *,
f'R%
#__inference__traced_restore_1691546��
�
�
F__inference_jojo_n8_r72x88_a0.7_b4_g2.2_strides2_layer_call_fn_1689955

inputs!
unknown:
	unknown_0:#
	unknown_1:#
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:#
	unknown_8:#
	unknown_9:1

unknown_10:1

unknown_11:1

unknown_12:1

unknown_13:1

unknown_14:1$

unknown_15:1$

unknown_16:1J

unknown_17:J

unknown_18:J

unknown_19:J

unknown_20:J

unknown_21:J$

unknown_22:J$

unknown_23:Jc

unknown_24:c

unknown_25:c

unknown_26:c

unknown_27:c

unknown_28:c

unknown_29:c

unknown_30:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30*,
Tin%
#2!*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*:
_read_only_resource_inputs

 *0
config_proto 

CPU

GPU2*0J 8� *j
feRc
a__inference_jojo_n8_r72x88_a0.7_b4_g2.2_strides2_layer_call_and_return_conditional_losses_1689408o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*n
_input_shapes]
[:���������HX: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������HX
 
_user_specified_nameinputs
�	
�
4__inference_CBlock_3_BatchNorm_layer_call_fn_1690537

inputs
unknown:J
	unknown_0:J
	unknown_1:J
	unknown_2:J
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������J*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *X
fSRQ
O__inference_CBlock_3_BatchNorm_layer_call_and_return_conditional_losses_1688976�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������J`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������J: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������J
 
_user_specified_nameinputs
��
�X
 __inference__traced_save_1691269
file_prefix=
#read_disablecopyonread_conv0_kernel:1
#read_1_disablecopyonread_conv0_bias:V
<read_2_disablecopyonread_cblock_1_sepconv2d_depthwise_kernel:V
<read_3_disablecopyonread_cblock_1_sepconv2d_pointwise_kernel:>
0read_4_disablecopyonread_cblock_1_sepconv2d_bias:?
1read_5_disablecopyonread_cblock_1_batchnorm_gamma:>
0read_6_disablecopyonread_cblock_1_batchnorm_beta:E
7read_7_disablecopyonread_cblock_1_batchnorm_moving_mean:I
;read_8_disablecopyonread_cblock_1_batchnorm_moving_variance:V
<read_9_disablecopyonread_cblock_2_sepconv2d_depthwise_kernel:W
=read_10_disablecopyonread_cblock_2_sepconv2d_pointwise_kernel:1?
1read_11_disablecopyonread_cblock_2_sepconv2d_bias:1@
2read_12_disablecopyonread_cblock_2_batchnorm_gamma:1?
1read_13_disablecopyonread_cblock_2_batchnorm_beta:1F
8read_14_disablecopyonread_cblock_2_batchnorm_moving_mean:1J
<read_15_disablecopyonread_cblock_2_batchnorm_moving_variance:1W
=read_16_disablecopyonread_cblock_3_sepconv2d_depthwise_kernel:1W
=read_17_disablecopyonread_cblock_3_sepconv2d_pointwise_kernel:1J?
1read_18_disablecopyonread_cblock_3_sepconv2d_bias:J@
2read_19_disablecopyonread_cblock_3_batchnorm_gamma:J?
1read_20_disablecopyonread_cblock_3_batchnorm_beta:JF
8read_21_disablecopyonread_cblock_3_batchnorm_moving_mean:JJ
<read_22_disablecopyonread_cblock_3_batchnorm_moving_variance:JW
=read_23_disablecopyonread_cblock_4_sepconv2d_depthwise_kernel:JW
=read_24_disablecopyonread_cblock_4_sepconv2d_pointwise_kernel:Jc?
1read_25_disablecopyonread_cblock_4_sepconv2d_bias:c@
2read_26_disablecopyonread_cblock_4_batchnorm_gamma:c?
1read_27_disablecopyonread_cblock_4_batchnorm_beta:cF
8read_28_disablecopyonread_cblock_4_batchnorm_moving_mean:cJ
<read_29_disablecopyonread_cblock_4_batchnorm_moving_variance:c:
(read_30_disablecopyonread_dense_8_kernel:c4
&read_31_disablecopyonread_dense_8_bias:-
#read_32_disablecopyonread_adam_iter:	 /
%read_33_disablecopyonread_adam_beta_1: /
%read_34_disablecopyonread_adam_beta_2: .
$read_35_disablecopyonread_adam_decay: 6
,read_36_disablecopyonread_adam_learning_rate: +
!read_37_disablecopyonread_total_1: +
!read_38_disablecopyonread_count_1: )
read_39_disablecopyonread_total: )
read_40_disablecopyonread_count: G
-read_41_disablecopyonread_adam_conv0_kernel_m:9
+read_42_disablecopyonread_adam_conv0_bias_m:^
Dread_43_disablecopyonread_adam_cblock_1_sepconv2d_depthwise_kernel_m:^
Dread_44_disablecopyonread_adam_cblock_1_sepconv2d_pointwise_kernel_m:F
8read_45_disablecopyonread_adam_cblock_1_sepconv2d_bias_m:G
9read_46_disablecopyonread_adam_cblock_1_batchnorm_gamma_m:F
8read_47_disablecopyonread_adam_cblock_1_batchnorm_beta_m:^
Dread_48_disablecopyonread_adam_cblock_2_sepconv2d_depthwise_kernel_m:^
Dread_49_disablecopyonread_adam_cblock_2_sepconv2d_pointwise_kernel_m:1F
8read_50_disablecopyonread_adam_cblock_2_sepconv2d_bias_m:1G
9read_51_disablecopyonread_adam_cblock_2_batchnorm_gamma_m:1F
8read_52_disablecopyonread_adam_cblock_2_batchnorm_beta_m:1^
Dread_53_disablecopyonread_adam_cblock_3_sepconv2d_depthwise_kernel_m:1^
Dread_54_disablecopyonread_adam_cblock_3_sepconv2d_pointwise_kernel_m:1JF
8read_55_disablecopyonread_adam_cblock_3_sepconv2d_bias_m:JG
9read_56_disablecopyonread_adam_cblock_3_batchnorm_gamma_m:JF
8read_57_disablecopyonread_adam_cblock_3_batchnorm_beta_m:J^
Dread_58_disablecopyonread_adam_cblock_4_sepconv2d_depthwise_kernel_m:J^
Dread_59_disablecopyonread_adam_cblock_4_sepconv2d_pointwise_kernel_m:JcF
8read_60_disablecopyonread_adam_cblock_4_sepconv2d_bias_m:cG
9read_61_disablecopyonread_adam_cblock_4_batchnorm_gamma_m:cF
8read_62_disablecopyonread_adam_cblock_4_batchnorm_beta_m:cA
/read_63_disablecopyonread_adam_dense_8_kernel_m:c;
-read_64_disablecopyonread_adam_dense_8_bias_m:G
-read_65_disablecopyonread_adam_conv0_kernel_v:9
+read_66_disablecopyonread_adam_conv0_bias_v:^
Dread_67_disablecopyonread_adam_cblock_1_sepconv2d_depthwise_kernel_v:^
Dread_68_disablecopyonread_adam_cblock_1_sepconv2d_pointwise_kernel_v:F
8read_69_disablecopyonread_adam_cblock_1_sepconv2d_bias_v:G
9read_70_disablecopyonread_adam_cblock_1_batchnorm_gamma_v:F
8read_71_disablecopyonread_adam_cblock_1_batchnorm_beta_v:^
Dread_72_disablecopyonread_adam_cblock_2_sepconv2d_depthwise_kernel_v:^
Dread_73_disablecopyonread_adam_cblock_2_sepconv2d_pointwise_kernel_v:1F
8read_74_disablecopyonread_adam_cblock_2_sepconv2d_bias_v:1G
9read_75_disablecopyonread_adam_cblock_2_batchnorm_gamma_v:1F
8read_76_disablecopyonread_adam_cblock_2_batchnorm_beta_v:1^
Dread_77_disablecopyonread_adam_cblock_3_sepconv2d_depthwise_kernel_v:1^
Dread_78_disablecopyonread_adam_cblock_3_sepconv2d_pointwise_kernel_v:1JF
8read_79_disablecopyonread_adam_cblock_3_sepconv2d_bias_v:JG
9read_80_disablecopyonread_adam_cblock_3_batchnorm_gamma_v:JF
8read_81_disablecopyonread_adam_cblock_3_batchnorm_beta_v:J^
Dread_82_disablecopyonread_adam_cblock_4_sepconv2d_depthwise_kernel_v:J^
Dread_83_disablecopyonread_adam_cblock_4_sepconv2d_pointwise_kernel_v:JcF
8read_84_disablecopyonread_adam_cblock_4_sepconv2d_bias_v:cG
9read_85_disablecopyonread_adam_cblock_4_batchnorm_gamma_v:cF
8read_86_disablecopyonread_adam_cblock_4_batchnorm_beta_v:cA
/read_87_disablecopyonread_adam_dense_8_kernel_v:c;
-read_88_disablecopyonread_adam_dense_8_bias_v:
savev2_const
identity_179��MergeV2Checkpoints�Read/DisableCopyOnRead�Read/ReadVariableOp�Read_1/DisableCopyOnRead�Read_1/ReadVariableOp�Read_10/DisableCopyOnRead�Read_10/ReadVariableOp�Read_11/DisableCopyOnRead�Read_11/ReadVariableOp�Read_12/DisableCopyOnRead�Read_12/ReadVariableOp�Read_13/DisableCopyOnRead�Read_13/ReadVariableOp�Read_14/DisableCopyOnRead�Read_14/ReadVariableOp�Read_15/DisableCopyOnRead�Read_15/ReadVariableOp�Read_16/DisableCopyOnRead�Read_16/ReadVariableOp�Read_17/DisableCopyOnRead�Read_17/ReadVariableOp�Read_18/DisableCopyOnRead�Read_18/ReadVariableOp�Read_19/DisableCopyOnRead�Read_19/ReadVariableOp�Read_2/DisableCopyOnRead�Read_2/ReadVariableOp�Read_20/DisableCopyOnRead�Read_20/ReadVariableOp�Read_21/DisableCopyOnRead�Read_21/ReadVariableOp�Read_22/DisableCopyOnRead�Read_22/ReadVariableOp�Read_23/DisableCopyOnRead�Read_23/ReadVariableOp�Read_24/DisableCopyOnRead�Read_24/ReadVariableOp�Read_25/DisableCopyOnRead�Read_25/ReadVariableOp�Read_26/DisableCopyOnRead�Read_26/ReadVariableOp�Read_27/DisableCopyOnRead�Read_27/ReadVariableOp�Read_28/DisableCopyOnRead�Read_28/ReadVariableOp�Read_29/DisableCopyOnRead�Read_29/ReadVariableOp�Read_3/DisableCopyOnRead�Read_3/ReadVariableOp�Read_30/DisableCopyOnRead�Read_30/ReadVariableOp�Read_31/DisableCopyOnRead�Read_31/ReadVariableOp�Read_32/DisableCopyOnRead�Read_32/ReadVariableOp�Read_33/DisableCopyOnRead�Read_33/ReadVariableOp�Read_34/DisableCopyOnRead�Read_34/ReadVariableOp�Read_35/DisableCopyOnRead�Read_35/ReadVariableOp�Read_36/DisableCopyOnRead�Read_36/ReadVariableOp�Read_37/DisableCopyOnRead�Read_37/ReadVariableOp�Read_38/DisableCopyOnRead�Read_38/ReadVariableOp�Read_39/DisableCopyOnRead�Read_39/ReadVariableOp�Read_4/DisableCopyOnRead�Read_4/ReadVariableOp�Read_40/DisableCopyOnRead�Read_40/ReadVariableOp�Read_41/DisableCopyOnRead�Read_41/ReadVariableOp�Read_42/DisableCopyOnRead�Read_42/ReadVariableOp�Read_43/DisableCopyOnRead�Read_43/ReadVariableOp�Read_44/DisableCopyOnRead�Read_44/ReadVariableOp�Read_45/DisableCopyOnRead�Read_45/ReadVariableOp�Read_46/DisableCopyOnRead�Read_46/ReadVariableOp�Read_47/DisableCopyOnRead�Read_47/ReadVariableOp�Read_48/DisableCopyOnRead�Read_48/ReadVariableOp�Read_49/DisableCopyOnRead�Read_49/ReadVariableOp�Read_5/DisableCopyOnRead�Read_5/ReadVariableOp�Read_50/DisableCopyOnRead�Read_50/ReadVariableOp�Read_51/DisableCopyOnRead�Read_51/ReadVariableOp�Read_52/DisableCopyOnRead�Read_52/ReadVariableOp�Read_53/DisableCopyOnRead�Read_53/ReadVariableOp�Read_54/DisableCopyOnRead�Read_54/ReadVariableOp�Read_55/DisableCopyOnRead�Read_55/ReadVariableOp�Read_56/DisableCopyOnRead�Read_56/ReadVariableOp�Read_57/DisableCopyOnRead�Read_57/ReadVariableOp�Read_58/DisableCopyOnRead�Read_58/ReadVariableOp�Read_59/DisableCopyOnRead�Read_59/ReadVariableOp�Read_6/DisableCopyOnRead�Read_6/ReadVariableOp�Read_60/DisableCopyOnRead�Read_60/ReadVariableOp�Read_61/DisableCopyOnRead�Read_61/ReadVariableOp�Read_62/DisableCopyOnRead�Read_62/ReadVariableOp�Read_63/DisableCopyOnRead�Read_63/ReadVariableOp�Read_64/DisableCopyOnRead�Read_64/ReadVariableOp�Read_65/DisableCopyOnRead�Read_65/ReadVariableOp�Read_66/DisableCopyOnRead�Read_66/ReadVariableOp�Read_67/DisableCopyOnRead�Read_67/ReadVariableOp�Read_68/DisableCopyOnRead�Read_68/ReadVariableOp�Read_69/DisableCopyOnRead�Read_69/ReadVariableOp�Read_7/DisableCopyOnRead�Read_7/ReadVariableOp�Read_70/DisableCopyOnRead�Read_70/ReadVariableOp�Read_71/DisableCopyOnRead�Read_71/ReadVariableOp�Read_72/DisableCopyOnRead�Read_72/ReadVariableOp�Read_73/DisableCopyOnRead�Read_73/ReadVariableOp�Read_74/DisableCopyOnRead�Read_74/ReadVariableOp�Read_75/DisableCopyOnRead�Read_75/ReadVariableOp�Read_76/DisableCopyOnRead�Read_76/ReadVariableOp�Read_77/DisableCopyOnRead�Read_77/ReadVariableOp�Read_78/DisableCopyOnRead�Read_78/ReadVariableOp�Read_79/DisableCopyOnRead�Read_79/ReadVariableOp�Read_8/DisableCopyOnRead�Read_8/ReadVariableOp�Read_80/DisableCopyOnRead�Read_80/ReadVariableOp�Read_81/DisableCopyOnRead�Read_81/ReadVariableOp�Read_82/DisableCopyOnRead�Read_82/ReadVariableOp�Read_83/DisableCopyOnRead�Read_83/ReadVariableOp�Read_84/DisableCopyOnRead�Read_84/ReadVariableOp�Read_85/DisableCopyOnRead�Read_85/ReadVariableOp�Read_86/DisableCopyOnRead�Read_86/ReadVariableOp�Read_87/DisableCopyOnRead�Read_87/ReadVariableOp�Read_88/DisableCopyOnRead�Read_88/ReadVariableOp�Read_9/DisableCopyOnRead�Read_9/ReadVariableOpw
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
: u
Read/DisableCopyOnReadDisableCopyOnRead#read_disablecopyonread_conv0_kernel"/device:CPU:0*
_output_shapes
 �
Read/ReadVariableOpReadVariableOp#read_disablecopyonread_conv0_kernel^Read/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:*
dtype0q
IdentityIdentityRead/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:i

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0*&
_output_shapes
:w
Read_1/DisableCopyOnReadDisableCopyOnRead#read_1_disablecopyonread_conv0_bias"/device:CPU:0*
_output_shapes
 �
Read_1/ReadVariableOpReadVariableOp#read_1_disablecopyonread_conv0_bias^Read_1/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0i

Identity_2IdentityRead_1/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:_

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_2/DisableCopyOnReadDisableCopyOnRead<read_2_disablecopyonread_cblock_1_sepconv2d_depthwise_kernel"/device:CPU:0*
_output_shapes
 �
Read_2/ReadVariableOpReadVariableOp<read_2_disablecopyonread_cblock_1_sepconv2d_depthwise_kernel^Read_2/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:*
dtype0u

Identity_4IdentityRead_2/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:k

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0*&
_output_shapes
:�
Read_3/DisableCopyOnReadDisableCopyOnRead<read_3_disablecopyonread_cblock_1_sepconv2d_pointwise_kernel"/device:CPU:0*
_output_shapes
 �
Read_3/ReadVariableOpReadVariableOp<read_3_disablecopyonread_cblock_1_sepconv2d_pointwise_kernel^Read_3/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:*
dtype0u

Identity_6IdentityRead_3/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:k

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*&
_output_shapes
:�
Read_4/DisableCopyOnReadDisableCopyOnRead0read_4_disablecopyonread_cblock_1_sepconv2d_bias"/device:CPU:0*
_output_shapes
 �
Read_4/ReadVariableOpReadVariableOp0read_4_disablecopyonread_cblock_1_sepconv2d_bias^Read_4/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0i

Identity_8IdentityRead_4/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:_

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_5/DisableCopyOnReadDisableCopyOnRead1read_5_disablecopyonread_cblock_1_batchnorm_gamma"/device:CPU:0*
_output_shapes
 �
Read_5/ReadVariableOpReadVariableOp1read_5_disablecopyonread_cblock_1_batchnorm_gamma^Read_5/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0j
Identity_10IdentityRead_5/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_6/DisableCopyOnReadDisableCopyOnRead0read_6_disablecopyonread_cblock_1_batchnorm_beta"/device:CPU:0*
_output_shapes
 �
Read_6/ReadVariableOpReadVariableOp0read_6_disablecopyonread_cblock_1_batchnorm_beta^Read_6/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0j
Identity_12IdentityRead_6/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_7/DisableCopyOnReadDisableCopyOnRead7read_7_disablecopyonread_cblock_1_batchnorm_moving_mean"/device:CPU:0*
_output_shapes
 �
Read_7/ReadVariableOpReadVariableOp7read_7_disablecopyonread_cblock_1_batchnorm_moving_mean^Read_7/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0j
Identity_14IdentityRead_7/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_8/DisableCopyOnReadDisableCopyOnRead;read_8_disablecopyonread_cblock_1_batchnorm_moving_variance"/device:CPU:0*
_output_shapes
 �
Read_8/ReadVariableOpReadVariableOp;read_8_disablecopyonread_cblock_1_batchnorm_moving_variance^Read_8/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0j
Identity_16IdentityRead_8/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_9/DisableCopyOnReadDisableCopyOnRead<read_9_disablecopyonread_cblock_2_sepconv2d_depthwise_kernel"/device:CPU:0*
_output_shapes
 �
Read_9/ReadVariableOpReadVariableOp<read_9_disablecopyonread_cblock_2_sepconv2d_depthwise_kernel^Read_9/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:*
dtype0v
Identity_18IdentityRead_9/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:m
Identity_19IdentityIdentity_18:output:0"/device:CPU:0*
T0*&
_output_shapes
:�
Read_10/DisableCopyOnReadDisableCopyOnRead=read_10_disablecopyonread_cblock_2_sepconv2d_pointwise_kernel"/device:CPU:0*
_output_shapes
 �
Read_10/ReadVariableOpReadVariableOp=read_10_disablecopyonread_cblock_2_sepconv2d_pointwise_kernel^Read_10/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:1*
dtype0w
Identity_20IdentityRead_10/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:1m
Identity_21IdentityIdentity_20:output:0"/device:CPU:0*
T0*&
_output_shapes
:1�
Read_11/DisableCopyOnReadDisableCopyOnRead1read_11_disablecopyonread_cblock_2_sepconv2d_bias"/device:CPU:0*
_output_shapes
 �
Read_11/ReadVariableOpReadVariableOp1read_11_disablecopyonread_cblock_2_sepconv2d_bias^Read_11/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:1*
dtype0k
Identity_22IdentityRead_11/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:1a
Identity_23IdentityIdentity_22:output:0"/device:CPU:0*
T0*
_output_shapes
:1�
Read_12/DisableCopyOnReadDisableCopyOnRead2read_12_disablecopyonread_cblock_2_batchnorm_gamma"/device:CPU:0*
_output_shapes
 �
Read_12/ReadVariableOpReadVariableOp2read_12_disablecopyonread_cblock_2_batchnorm_gamma^Read_12/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:1*
dtype0k
Identity_24IdentityRead_12/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:1a
Identity_25IdentityIdentity_24:output:0"/device:CPU:0*
T0*
_output_shapes
:1�
Read_13/DisableCopyOnReadDisableCopyOnRead1read_13_disablecopyonread_cblock_2_batchnorm_beta"/device:CPU:0*
_output_shapes
 �
Read_13/ReadVariableOpReadVariableOp1read_13_disablecopyonread_cblock_2_batchnorm_beta^Read_13/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:1*
dtype0k
Identity_26IdentityRead_13/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:1a
Identity_27IdentityIdentity_26:output:0"/device:CPU:0*
T0*
_output_shapes
:1�
Read_14/DisableCopyOnReadDisableCopyOnRead8read_14_disablecopyonread_cblock_2_batchnorm_moving_mean"/device:CPU:0*
_output_shapes
 �
Read_14/ReadVariableOpReadVariableOp8read_14_disablecopyonread_cblock_2_batchnorm_moving_mean^Read_14/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:1*
dtype0k
Identity_28IdentityRead_14/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:1a
Identity_29IdentityIdentity_28:output:0"/device:CPU:0*
T0*
_output_shapes
:1�
Read_15/DisableCopyOnReadDisableCopyOnRead<read_15_disablecopyonread_cblock_2_batchnorm_moving_variance"/device:CPU:0*
_output_shapes
 �
Read_15/ReadVariableOpReadVariableOp<read_15_disablecopyonread_cblock_2_batchnorm_moving_variance^Read_15/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:1*
dtype0k
Identity_30IdentityRead_15/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:1a
Identity_31IdentityIdentity_30:output:0"/device:CPU:0*
T0*
_output_shapes
:1�
Read_16/DisableCopyOnReadDisableCopyOnRead=read_16_disablecopyonread_cblock_3_sepconv2d_depthwise_kernel"/device:CPU:0*
_output_shapes
 �
Read_16/ReadVariableOpReadVariableOp=read_16_disablecopyonread_cblock_3_sepconv2d_depthwise_kernel^Read_16/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:1*
dtype0w
Identity_32IdentityRead_16/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:1m
Identity_33IdentityIdentity_32:output:0"/device:CPU:0*
T0*&
_output_shapes
:1�
Read_17/DisableCopyOnReadDisableCopyOnRead=read_17_disablecopyonread_cblock_3_sepconv2d_pointwise_kernel"/device:CPU:0*
_output_shapes
 �
Read_17/ReadVariableOpReadVariableOp=read_17_disablecopyonread_cblock_3_sepconv2d_pointwise_kernel^Read_17/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:1J*
dtype0w
Identity_34IdentityRead_17/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:1Jm
Identity_35IdentityIdentity_34:output:0"/device:CPU:0*
T0*&
_output_shapes
:1J�
Read_18/DisableCopyOnReadDisableCopyOnRead1read_18_disablecopyonread_cblock_3_sepconv2d_bias"/device:CPU:0*
_output_shapes
 �
Read_18/ReadVariableOpReadVariableOp1read_18_disablecopyonread_cblock_3_sepconv2d_bias^Read_18/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:J*
dtype0k
Identity_36IdentityRead_18/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:Ja
Identity_37IdentityIdentity_36:output:0"/device:CPU:0*
T0*
_output_shapes
:J�
Read_19/DisableCopyOnReadDisableCopyOnRead2read_19_disablecopyonread_cblock_3_batchnorm_gamma"/device:CPU:0*
_output_shapes
 �
Read_19/ReadVariableOpReadVariableOp2read_19_disablecopyonread_cblock_3_batchnorm_gamma^Read_19/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:J*
dtype0k
Identity_38IdentityRead_19/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:Ja
Identity_39IdentityIdentity_38:output:0"/device:CPU:0*
T0*
_output_shapes
:J�
Read_20/DisableCopyOnReadDisableCopyOnRead1read_20_disablecopyonread_cblock_3_batchnorm_beta"/device:CPU:0*
_output_shapes
 �
Read_20/ReadVariableOpReadVariableOp1read_20_disablecopyonread_cblock_3_batchnorm_beta^Read_20/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:J*
dtype0k
Identity_40IdentityRead_20/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:Ja
Identity_41IdentityIdentity_40:output:0"/device:CPU:0*
T0*
_output_shapes
:J�
Read_21/DisableCopyOnReadDisableCopyOnRead8read_21_disablecopyonread_cblock_3_batchnorm_moving_mean"/device:CPU:0*
_output_shapes
 �
Read_21/ReadVariableOpReadVariableOp8read_21_disablecopyonread_cblock_3_batchnorm_moving_mean^Read_21/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:J*
dtype0k
Identity_42IdentityRead_21/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:Ja
Identity_43IdentityIdentity_42:output:0"/device:CPU:0*
T0*
_output_shapes
:J�
Read_22/DisableCopyOnReadDisableCopyOnRead<read_22_disablecopyonread_cblock_3_batchnorm_moving_variance"/device:CPU:0*
_output_shapes
 �
Read_22/ReadVariableOpReadVariableOp<read_22_disablecopyonread_cblock_3_batchnorm_moving_variance^Read_22/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:J*
dtype0k
Identity_44IdentityRead_22/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:Ja
Identity_45IdentityIdentity_44:output:0"/device:CPU:0*
T0*
_output_shapes
:J�
Read_23/DisableCopyOnReadDisableCopyOnRead=read_23_disablecopyonread_cblock_4_sepconv2d_depthwise_kernel"/device:CPU:0*
_output_shapes
 �
Read_23/ReadVariableOpReadVariableOp=read_23_disablecopyonread_cblock_4_sepconv2d_depthwise_kernel^Read_23/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:J*
dtype0w
Identity_46IdentityRead_23/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:Jm
Identity_47IdentityIdentity_46:output:0"/device:CPU:0*
T0*&
_output_shapes
:J�
Read_24/DisableCopyOnReadDisableCopyOnRead=read_24_disablecopyonread_cblock_4_sepconv2d_pointwise_kernel"/device:CPU:0*
_output_shapes
 �
Read_24/ReadVariableOpReadVariableOp=read_24_disablecopyonread_cblock_4_sepconv2d_pointwise_kernel^Read_24/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:Jc*
dtype0w
Identity_48IdentityRead_24/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:Jcm
Identity_49IdentityIdentity_48:output:0"/device:CPU:0*
T0*&
_output_shapes
:Jc�
Read_25/DisableCopyOnReadDisableCopyOnRead1read_25_disablecopyonread_cblock_4_sepconv2d_bias"/device:CPU:0*
_output_shapes
 �
Read_25/ReadVariableOpReadVariableOp1read_25_disablecopyonread_cblock_4_sepconv2d_bias^Read_25/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:c*
dtype0k
Identity_50IdentityRead_25/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:ca
Identity_51IdentityIdentity_50:output:0"/device:CPU:0*
T0*
_output_shapes
:c�
Read_26/DisableCopyOnReadDisableCopyOnRead2read_26_disablecopyonread_cblock_4_batchnorm_gamma"/device:CPU:0*
_output_shapes
 �
Read_26/ReadVariableOpReadVariableOp2read_26_disablecopyonread_cblock_4_batchnorm_gamma^Read_26/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:c*
dtype0k
Identity_52IdentityRead_26/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:ca
Identity_53IdentityIdentity_52:output:0"/device:CPU:0*
T0*
_output_shapes
:c�
Read_27/DisableCopyOnReadDisableCopyOnRead1read_27_disablecopyonread_cblock_4_batchnorm_beta"/device:CPU:0*
_output_shapes
 �
Read_27/ReadVariableOpReadVariableOp1read_27_disablecopyonread_cblock_4_batchnorm_beta^Read_27/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:c*
dtype0k
Identity_54IdentityRead_27/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:ca
Identity_55IdentityIdentity_54:output:0"/device:CPU:0*
T0*
_output_shapes
:c�
Read_28/DisableCopyOnReadDisableCopyOnRead8read_28_disablecopyonread_cblock_4_batchnorm_moving_mean"/device:CPU:0*
_output_shapes
 �
Read_28/ReadVariableOpReadVariableOp8read_28_disablecopyonread_cblock_4_batchnorm_moving_mean^Read_28/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:c*
dtype0k
Identity_56IdentityRead_28/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:ca
Identity_57IdentityIdentity_56:output:0"/device:CPU:0*
T0*
_output_shapes
:c�
Read_29/DisableCopyOnReadDisableCopyOnRead<read_29_disablecopyonread_cblock_4_batchnorm_moving_variance"/device:CPU:0*
_output_shapes
 �
Read_29/ReadVariableOpReadVariableOp<read_29_disablecopyonread_cblock_4_batchnorm_moving_variance^Read_29/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:c*
dtype0k
Identity_58IdentityRead_29/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:ca
Identity_59IdentityIdentity_58:output:0"/device:CPU:0*
T0*
_output_shapes
:c}
Read_30/DisableCopyOnReadDisableCopyOnRead(read_30_disablecopyonread_dense_8_kernel"/device:CPU:0*
_output_shapes
 �
Read_30/ReadVariableOpReadVariableOp(read_30_disablecopyonread_dense_8_kernel^Read_30/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:c*
dtype0o
Identity_60IdentityRead_30/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:ce
Identity_61IdentityIdentity_60:output:0"/device:CPU:0*
T0*
_output_shapes

:c{
Read_31/DisableCopyOnReadDisableCopyOnRead&read_31_disablecopyonread_dense_8_bias"/device:CPU:0*
_output_shapes
 �
Read_31/ReadVariableOpReadVariableOp&read_31_disablecopyonread_dense_8_bias^Read_31/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_62IdentityRead_31/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_63IdentityIdentity_62:output:0"/device:CPU:0*
T0*
_output_shapes
:x
Read_32/DisableCopyOnReadDisableCopyOnRead#read_32_disablecopyonread_adam_iter"/device:CPU:0*
_output_shapes
 �
Read_32/ReadVariableOpReadVariableOp#read_32_disablecopyonread_adam_iter^Read_32/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0	g
Identity_64IdentityRead_32/ReadVariableOp:value:0"/device:CPU:0*
T0	*
_output_shapes
: ]
Identity_65IdentityIdentity_64:output:0"/device:CPU:0*
T0	*
_output_shapes
: z
Read_33/DisableCopyOnReadDisableCopyOnRead%read_33_disablecopyonread_adam_beta_1"/device:CPU:0*
_output_shapes
 �
Read_33/ReadVariableOpReadVariableOp%read_33_disablecopyonread_adam_beta_1^Read_33/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_66IdentityRead_33/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_67IdentityIdentity_66:output:0"/device:CPU:0*
T0*
_output_shapes
: z
Read_34/DisableCopyOnReadDisableCopyOnRead%read_34_disablecopyonread_adam_beta_2"/device:CPU:0*
_output_shapes
 �
Read_34/ReadVariableOpReadVariableOp%read_34_disablecopyonread_adam_beta_2^Read_34/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_68IdentityRead_34/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_69IdentityIdentity_68:output:0"/device:CPU:0*
T0*
_output_shapes
: y
Read_35/DisableCopyOnReadDisableCopyOnRead$read_35_disablecopyonread_adam_decay"/device:CPU:0*
_output_shapes
 �
Read_35/ReadVariableOpReadVariableOp$read_35_disablecopyonread_adam_decay^Read_35/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_70IdentityRead_35/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_71IdentityIdentity_70:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_36/DisableCopyOnReadDisableCopyOnRead,read_36_disablecopyonread_adam_learning_rate"/device:CPU:0*
_output_shapes
 �
Read_36/ReadVariableOpReadVariableOp,read_36_disablecopyonread_adam_learning_rate^Read_36/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_72IdentityRead_36/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_73IdentityIdentity_72:output:0"/device:CPU:0*
T0*
_output_shapes
: v
Read_37/DisableCopyOnReadDisableCopyOnRead!read_37_disablecopyonread_total_1"/device:CPU:0*
_output_shapes
 �
Read_37/ReadVariableOpReadVariableOp!read_37_disablecopyonread_total_1^Read_37/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_74IdentityRead_37/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_75IdentityIdentity_74:output:0"/device:CPU:0*
T0*
_output_shapes
: v
Read_38/DisableCopyOnReadDisableCopyOnRead!read_38_disablecopyonread_count_1"/device:CPU:0*
_output_shapes
 �
Read_38/ReadVariableOpReadVariableOp!read_38_disablecopyonread_count_1^Read_38/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_76IdentityRead_38/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_77IdentityIdentity_76:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_39/DisableCopyOnReadDisableCopyOnReadread_39_disablecopyonread_total"/device:CPU:0*
_output_shapes
 �
Read_39/ReadVariableOpReadVariableOpread_39_disablecopyonread_total^Read_39/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_78IdentityRead_39/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_79IdentityIdentity_78:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_40/DisableCopyOnReadDisableCopyOnReadread_40_disablecopyonread_count"/device:CPU:0*
_output_shapes
 �
Read_40/ReadVariableOpReadVariableOpread_40_disablecopyonread_count^Read_40/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_80IdentityRead_40/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_81IdentityIdentity_80:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_41/DisableCopyOnReadDisableCopyOnRead-read_41_disablecopyonread_adam_conv0_kernel_m"/device:CPU:0*
_output_shapes
 �
Read_41/ReadVariableOpReadVariableOp-read_41_disablecopyonread_adam_conv0_kernel_m^Read_41/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:*
dtype0w
Identity_82IdentityRead_41/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:m
Identity_83IdentityIdentity_82:output:0"/device:CPU:0*
T0*&
_output_shapes
:�
Read_42/DisableCopyOnReadDisableCopyOnRead+read_42_disablecopyonread_adam_conv0_bias_m"/device:CPU:0*
_output_shapes
 �
Read_42/ReadVariableOpReadVariableOp+read_42_disablecopyonread_adam_conv0_bias_m^Read_42/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_84IdentityRead_42/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_85IdentityIdentity_84:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_43/DisableCopyOnReadDisableCopyOnReadDread_43_disablecopyonread_adam_cblock_1_sepconv2d_depthwise_kernel_m"/device:CPU:0*
_output_shapes
 �
Read_43/ReadVariableOpReadVariableOpDread_43_disablecopyonread_adam_cblock_1_sepconv2d_depthwise_kernel_m^Read_43/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:*
dtype0w
Identity_86IdentityRead_43/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:m
Identity_87IdentityIdentity_86:output:0"/device:CPU:0*
T0*&
_output_shapes
:�
Read_44/DisableCopyOnReadDisableCopyOnReadDread_44_disablecopyonread_adam_cblock_1_sepconv2d_pointwise_kernel_m"/device:CPU:0*
_output_shapes
 �
Read_44/ReadVariableOpReadVariableOpDread_44_disablecopyonread_adam_cblock_1_sepconv2d_pointwise_kernel_m^Read_44/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:*
dtype0w
Identity_88IdentityRead_44/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:m
Identity_89IdentityIdentity_88:output:0"/device:CPU:0*
T0*&
_output_shapes
:�
Read_45/DisableCopyOnReadDisableCopyOnRead8read_45_disablecopyonread_adam_cblock_1_sepconv2d_bias_m"/device:CPU:0*
_output_shapes
 �
Read_45/ReadVariableOpReadVariableOp8read_45_disablecopyonread_adam_cblock_1_sepconv2d_bias_m^Read_45/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_90IdentityRead_45/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_91IdentityIdentity_90:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_46/DisableCopyOnReadDisableCopyOnRead9read_46_disablecopyonread_adam_cblock_1_batchnorm_gamma_m"/device:CPU:0*
_output_shapes
 �
Read_46/ReadVariableOpReadVariableOp9read_46_disablecopyonread_adam_cblock_1_batchnorm_gamma_m^Read_46/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_92IdentityRead_46/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_93IdentityIdentity_92:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_47/DisableCopyOnReadDisableCopyOnRead8read_47_disablecopyonread_adam_cblock_1_batchnorm_beta_m"/device:CPU:0*
_output_shapes
 �
Read_47/ReadVariableOpReadVariableOp8read_47_disablecopyonread_adam_cblock_1_batchnorm_beta_m^Read_47/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_94IdentityRead_47/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_95IdentityIdentity_94:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_48/DisableCopyOnReadDisableCopyOnReadDread_48_disablecopyonread_adam_cblock_2_sepconv2d_depthwise_kernel_m"/device:CPU:0*
_output_shapes
 �
Read_48/ReadVariableOpReadVariableOpDread_48_disablecopyonread_adam_cblock_2_sepconv2d_depthwise_kernel_m^Read_48/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:*
dtype0w
Identity_96IdentityRead_48/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:m
Identity_97IdentityIdentity_96:output:0"/device:CPU:0*
T0*&
_output_shapes
:�
Read_49/DisableCopyOnReadDisableCopyOnReadDread_49_disablecopyonread_adam_cblock_2_sepconv2d_pointwise_kernel_m"/device:CPU:0*
_output_shapes
 �
Read_49/ReadVariableOpReadVariableOpDread_49_disablecopyonread_adam_cblock_2_sepconv2d_pointwise_kernel_m^Read_49/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:1*
dtype0w
Identity_98IdentityRead_49/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:1m
Identity_99IdentityIdentity_98:output:0"/device:CPU:0*
T0*&
_output_shapes
:1�
Read_50/DisableCopyOnReadDisableCopyOnRead8read_50_disablecopyonread_adam_cblock_2_sepconv2d_bias_m"/device:CPU:0*
_output_shapes
 �
Read_50/ReadVariableOpReadVariableOp8read_50_disablecopyonread_adam_cblock_2_sepconv2d_bias_m^Read_50/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:1*
dtype0l
Identity_100IdentityRead_50/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:1c
Identity_101IdentityIdentity_100:output:0"/device:CPU:0*
T0*
_output_shapes
:1�
Read_51/DisableCopyOnReadDisableCopyOnRead9read_51_disablecopyonread_adam_cblock_2_batchnorm_gamma_m"/device:CPU:0*
_output_shapes
 �
Read_51/ReadVariableOpReadVariableOp9read_51_disablecopyonread_adam_cblock_2_batchnorm_gamma_m^Read_51/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:1*
dtype0l
Identity_102IdentityRead_51/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:1c
Identity_103IdentityIdentity_102:output:0"/device:CPU:0*
T0*
_output_shapes
:1�
Read_52/DisableCopyOnReadDisableCopyOnRead8read_52_disablecopyonread_adam_cblock_2_batchnorm_beta_m"/device:CPU:0*
_output_shapes
 �
Read_52/ReadVariableOpReadVariableOp8read_52_disablecopyonread_adam_cblock_2_batchnorm_beta_m^Read_52/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:1*
dtype0l
Identity_104IdentityRead_52/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:1c
Identity_105IdentityIdentity_104:output:0"/device:CPU:0*
T0*
_output_shapes
:1�
Read_53/DisableCopyOnReadDisableCopyOnReadDread_53_disablecopyonread_adam_cblock_3_sepconv2d_depthwise_kernel_m"/device:CPU:0*
_output_shapes
 �
Read_53/ReadVariableOpReadVariableOpDread_53_disablecopyonread_adam_cblock_3_sepconv2d_depthwise_kernel_m^Read_53/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:1*
dtype0x
Identity_106IdentityRead_53/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:1o
Identity_107IdentityIdentity_106:output:0"/device:CPU:0*
T0*&
_output_shapes
:1�
Read_54/DisableCopyOnReadDisableCopyOnReadDread_54_disablecopyonread_adam_cblock_3_sepconv2d_pointwise_kernel_m"/device:CPU:0*
_output_shapes
 �
Read_54/ReadVariableOpReadVariableOpDread_54_disablecopyonread_adam_cblock_3_sepconv2d_pointwise_kernel_m^Read_54/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:1J*
dtype0x
Identity_108IdentityRead_54/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:1Jo
Identity_109IdentityIdentity_108:output:0"/device:CPU:0*
T0*&
_output_shapes
:1J�
Read_55/DisableCopyOnReadDisableCopyOnRead8read_55_disablecopyonread_adam_cblock_3_sepconv2d_bias_m"/device:CPU:0*
_output_shapes
 �
Read_55/ReadVariableOpReadVariableOp8read_55_disablecopyonread_adam_cblock_3_sepconv2d_bias_m^Read_55/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:J*
dtype0l
Identity_110IdentityRead_55/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:Jc
Identity_111IdentityIdentity_110:output:0"/device:CPU:0*
T0*
_output_shapes
:J�
Read_56/DisableCopyOnReadDisableCopyOnRead9read_56_disablecopyonread_adam_cblock_3_batchnorm_gamma_m"/device:CPU:0*
_output_shapes
 �
Read_56/ReadVariableOpReadVariableOp9read_56_disablecopyonread_adam_cblock_3_batchnorm_gamma_m^Read_56/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:J*
dtype0l
Identity_112IdentityRead_56/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:Jc
Identity_113IdentityIdentity_112:output:0"/device:CPU:0*
T0*
_output_shapes
:J�
Read_57/DisableCopyOnReadDisableCopyOnRead8read_57_disablecopyonread_adam_cblock_3_batchnorm_beta_m"/device:CPU:0*
_output_shapes
 �
Read_57/ReadVariableOpReadVariableOp8read_57_disablecopyonread_adam_cblock_3_batchnorm_beta_m^Read_57/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:J*
dtype0l
Identity_114IdentityRead_57/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:Jc
Identity_115IdentityIdentity_114:output:0"/device:CPU:0*
T0*
_output_shapes
:J�
Read_58/DisableCopyOnReadDisableCopyOnReadDread_58_disablecopyonread_adam_cblock_4_sepconv2d_depthwise_kernel_m"/device:CPU:0*
_output_shapes
 �
Read_58/ReadVariableOpReadVariableOpDread_58_disablecopyonread_adam_cblock_4_sepconv2d_depthwise_kernel_m^Read_58/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:J*
dtype0x
Identity_116IdentityRead_58/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:Jo
Identity_117IdentityIdentity_116:output:0"/device:CPU:0*
T0*&
_output_shapes
:J�
Read_59/DisableCopyOnReadDisableCopyOnReadDread_59_disablecopyonread_adam_cblock_4_sepconv2d_pointwise_kernel_m"/device:CPU:0*
_output_shapes
 �
Read_59/ReadVariableOpReadVariableOpDread_59_disablecopyonread_adam_cblock_4_sepconv2d_pointwise_kernel_m^Read_59/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:Jc*
dtype0x
Identity_118IdentityRead_59/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:Jco
Identity_119IdentityIdentity_118:output:0"/device:CPU:0*
T0*&
_output_shapes
:Jc�
Read_60/DisableCopyOnReadDisableCopyOnRead8read_60_disablecopyonread_adam_cblock_4_sepconv2d_bias_m"/device:CPU:0*
_output_shapes
 �
Read_60/ReadVariableOpReadVariableOp8read_60_disablecopyonread_adam_cblock_4_sepconv2d_bias_m^Read_60/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:c*
dtype0l
Identity_120IdentityRead_60/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:cc
Identity_121IdentityIdentity_120:output:0"/device:CPU:0*
T0*
_output_shapes
:c�
Read_61/DisableCopyOnReadDisableCopyOnRead9read_61_disablecopyonread_adam_cblock_4_batchnorm_gamma_m"/device:CPU:0*
_output_shapes
 �
Read_61/ReadVariableOpReadVariableOp9read_61_disablecopyonread_adam_cblock_4_batchnorm_gamma_m^Read_61/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:c*
dtype0l
Identity_122IdentityRead_61/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:cc
Identity_123IdentityIdentity_122:output:0"/device:CPU:0*
T0*
_output_shapes
:c�
Read_62/DisableCopyOnReadDisableCopyOnRead8read_62_disablecopyonread_adam_cblock_4_batchnorm_beta_m"/device:CPU:0*
_output_shapes
 �
Read_62/ReadVariableOpReadVariableOp8read_62_disablecopyonread_adam_cblock_4_batchnorm_beta_m^Read_62/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:c*
dtype0l
Identity_124IdentityRead_62/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:cc
Identity_125IdentityIdentity_124:output:0"/device:CPU:0*
T0*
_output_shapes
:c�
Read_63/DisableCopyOnReadDisableCopyOnRead/read_63_disablecopyonread_adam_dense_8_kernel_m"/device:CPU:0*
_output_shapes
 �
Read_63/ReadVariableOpReadVariableOp/read_63_disablecopyonread_adam_dense_8_kernel_m^Read_63/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:c*
dtype0p
Identity_126IdentityRead_63/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:cg
Identity_127IdentityIdentity_126:output:0"/device:CPU:0*
T0*
_output_shapes

:c�
Read_64/DisableCopyOnReadDisableCopyOnRead-read_64_disablecopyonread_adam_dense_8_bias_m"/device:CPU:0*
_output_shapes
 �
Read_64/ReadVariableOpReadVariableOp-read_64_disablecopyonread_adam_dense_8_bias_m^Read_64/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_128IdentityRead_64/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_129IdentityIdentity_128:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_65/DisableCopyOnReadDisableCopyOnRead-read_65_disablecopyonread_adam_conv0_kernel_v"/device:CPU:0*
_output_shapes
 �
Read_65/ReadVariableOpReadVariableOp-read_65_disablecopyonread_adam_conv0_kernel_v^Read_65/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:*
dtype0x
Identity_130IdentityRead_65/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:o
Identity_131IdentityIdentity_130:output:0"/device:CPU:0*
T0*&
_output_shapes
:�
Read_66/DisableCopyOnReadDisableCopyOnRead+read_66_disablecopyonread_adam_conv0_bias_v"/device:CPU:0*
_output_shapes
 �
Read_66/ReadVariableOpReadVariableOp+read_66_disablecopyonread_adam_conv0_bias_v^Read_66/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_132IdentityRead_66/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_133IdentityIdentity_132:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_67/DisableCopyOnReadDisableCopyOnReadDread_67_disablecopyonread_adam_cblock_1_sepconv2d_depthwise_kernel_v"/device:CPU:0*
_output_shapes
 �
Read_67/ReadVariableOpReadVariableOpDread_67_disablecopyonread_adam_cblock_1_sepconv2d_depthwise_kernel_v^Read_67/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:*
dtype0x
Identity_134IdentityRead_67/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:o
Identity_135IdentityIdentity_134:output:0"/device:CPU:0*
T0*&
_output_shapes
:�
Read_68/DisableCopyOnReadDisableCopyOnReadDread_68_disablecopyonread_adam_cblock_1_sepconv2d_pointwise_kernel_v"/device:CPU:0*
_output_shapes
 �
Read_68/ReadVariableOpReadVariableOpDread_68_disablecopyonread_adam_cblock_1_sepconv2d_pointwise_kernel_v^Read_68/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:*
dtype0x
Identity_136IdentityRead_68/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:o
Identity_137IdentityIdentity_136:output:0"/device:CPU:0*
T0*&
_output_shapes
:�
Read_69/DisableCopyOnReadDisableCopyOnRead8read_69_disablecopyonread_adam_cblock_1_sepconv2d_bias_v"/device:CPU:0*
_output_shapes
 �
Read_69/ReadVariableOpReadVariableOp8read_69_disablecopyonread_adam_cblock_1_sepconv2d_bias_v^Read_69/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_138IdentityRead_69/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_139IdentityIdentity_138:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_70/DisableCopyOnReadDisableCopyOnRead9read_70_disablecopyonread_adam_cblock_1_batchnorm_gamma_v"/device:CPU:0*
_output_shapes
 �
Read_70/ReadVariableOpReadVariableOp9read_70_disablecopyonread_adam_cblock_1_batchnorm_gamma_v^Read_70/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_140IdentityRead_70/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_141IdentityIdentity_140:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_71/DisableCopyOnReadDisableCopyOnRead8read_71_disablecopyonread_adam_cblock_1_batchnorm_beta_v"/device:CPU:0*
_output_shapes
 �
Read_71/ReadVariableOpReadVariableOp8read_71_disablecopyonread_adam_cblock_1_batchnorm_beta_v^Read_71/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_142IdentityRead_71/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_143IdentityIdentity_142:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_72/DisableCopyOnReadDisableCopyOnReadDread_72_disablecopyonread_adam_cblock_2_sepconv2d_depthwise_kernel_v"/device:CPU:0*
_output_shapes
 �
Read_72/ReadVariableOpReadVariableOpDread_72_disablecopyonread_adam_cblock_2_sepconv2d_depthwise_kernel_v^Read_72/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:*
dtype0x
Identity_144IdentityRead_72/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:o
Identity_145IdentityIdentity_144:output:0"/device:CPU:0*
T0*&
_output_shapes
:�
Read_73/DisableCopyOnReadDisableCopyOnReadDread_73_disablecopyonread_adam_cblock_2_sepconv2d_pointwise_kernel_v"/device:CPU:0*
_output_shapes
 �
Read_73/ReadVariableOpReadVariableOpDread_73_disablecopyonread_adam_cblock_2_sepconv2d_pointwise_kernel_v^Read_73/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:1*
dtype0x
Identity_146IdentityRead_73/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:1o
Identity_147IdentityIdentity_146:output:0"/device:CPU:0*
T0*&
_output_shapes
:1�
Read_74/DisableCopyOnReadDisableCopyOnRead8read_74_disablecopyonread_adam_cblock_2_sepconv2d_bias_v"/device:CPU:0*
_output_shapes
 �
Read_74/ReadVariableOpReadVariableOp8read_74_disablecopyonread_adam_cblock_2_sepconv2d_bias_v^Read_74/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:1*
dtype0l
Identity_148IdentityRead_74/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:1c
Identity_149IdentityIdentity_148:output:0"/device:CPU:0*
T0*
_output_shapes
:1�
Read_75/DisableCopyOnReadDisableCopyOnRead9read_75_disablecopyonread_adam_cblock_2_batchnorm_gamma_v"/device:CPU:0*
_output_shapes
 �
Read_75/ReadVariableOpReadVariableOp9read_75_disablecopyonread_adam_cblock_2_batchnorm_gamma_v^Read_75/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:1*
dtype0l
Identity_150IdentityRead_75/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:1c
Identity_151IdentityIdentity_150:output:0"/device:CPU:0*
T0*
_output_shapes
:1�
Read_76/DisableCopyOnReadDisableCopyOnRead8read_76_disablecopyonread_adam_cblock_2_batchnorm_beta_v"/device:CPU:0*
_output_shapes
 �
Read_76/ReadVariableOpReadVariableOp8read_76_disablecopyonread_adam_cblock_2_batchnorm_beta_v^Read_76/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:1*
dtype0l
Identity_152IdentityRead_76/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:1c
Identity_153IdentityIdentity_152:output:0"/device:CPU:0*
T0*
_output_shapes
:1�
Read_77/DisableCopyOnReadDisableCopyOnReadDread_77_disablecopyonread_adam_cblock_3_sepconv2d_depthwise_kernel_v"/device:CPU:0*
_output_shapes
 �
Read_77/ReadVariableOpReadVariableOpDread_77_disablecopyonread_adam_cblock_3_sepconv2d_depthwise_kernel_v^Read_77/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:1*
dtype0x
Identity_154IdentityRead_77/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:1o
Identity_155IdentityIdentity_154:output:0"/device:CPU:0*
T0*&
_output_shapes
:1�
Read_78/DisableCopyOnReadDisableCopyOnReadDread_78_disablecopyonread_adam_cblock_3_sepconv2d_pointwise_kernel_v"/device:CPU:0*
_output_shapes
 �
Read_78/ReadVariableOpReadVariableOpDread_78_disablecopyonread_adam_cblock_3_sepconv2d_pointwise_kernel_v^Read_78/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:1J*
dtype0x
Identity_156IdentityRead_78/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:1Jo
Identity_157IdentityIdentity_156:output:0"/device:CPU:0*
T0*&
_output_shapes
:1J�
Read_79/DisableCopyOnReadDisableCopyOnRead8read_79_disablecopyonread_adam_cblock_3_sepconv2d_bias_v"/device:CPU:0*
_output_shapes
 �
Read_79/ReadVariableOpReadVariableOp8read_79_disablecopyonread_adam_cblock_3_sepconv2d_bias_v^Read_79/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:J*
dtype0l
Identity_158IdentityRead_79/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:Jc
Identity_159IdentityIdentity_158:output:0"/device:CPU:0*
T0*
_output_shapes
:J�
Read_80/DisableCopyOnReadDisableCopyOnRead9read_80_disablecopyonread_adam_cblock_3_batchnorm_gamma_v"/device:CPU:0*
_output_shapes
 �
Read_80/ReadVariableOpReadVariableOp9read_80_disablecopyonread_adam_cblock_3_batchnorm_gamma_v^Read_80/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:J*
dtype0l
Identity_160IdentityRead_80/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:Jc
Identity_161IdentityIdentity_160:output:0"/device:CPU:0*
T0*
_output_shapes
:J�
Read_81/DisableCopyOnReadDisableCopyOnRead8read_81_disablecopyonread_adam_cblock_3_batchnorm_beta_v"/device:CPU:0*
_output_shapes
 �
Read_81/ReadVariableOpReadVariableOp8read_81_disablecopyonread_adam_cblock_3_batchnorm_beta_v^Read_81/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:J*
dtype0l
Identity_162IdentityRead_81/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:Jc
Identity_163IdentityIdentity_162:output:0"/device:CPU:0*
T0*
_output_shapes
:J�
Read_82/DisableCopyOnReadDisableCopyOnReadDread_82_disablecopyonread_adam_cblock_4_sepconv2d_depthwise_kernel_v"/device:CPU:0*
_output_shapes
 �
Read_82/ReadVariableOpReadVariableOpDread_82_disablecopyonread_adam_cblock_4_sepconv2d_depthwise_kernel_v^Read_82/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:J*
dtype0x
Identity_164IdentityRead_82/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:Jo
Identity_165IdentityIdentity_164:output:0"/device:CPU:0*
T0*&
_output_shapes
:J�
Read_83/DisableCopyOnReadDisableCopyOnReadDread_83_disablecopyonread_adam_cblock_4_sepconv2d_pointwise_kernel_v"/device:CPU:0*
_output_shapes
 �
Read_83/ReadVariableOpReadVariableOpDread_83_disablecopyonread_adam_cblock_4_sepconv2d_pointwise_kernel_v^Read_83/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:Jc*
dtype0x
Identity_166IdentityRead_83/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:Jco
Identity_167IdentityIdentity_166:output:0"/device:CPU:0*
T0*&
_output_shapes
:Jc�
Read_84/DisableCopyOnReadDisableCopyOnRead8read_84_disablecopyonread_adam_cblock_4_sepconv2d_bias_v"/device:CPU:0*
_output_shapes
 �
Read_84/ReadVariableOpReadVariableOp8read_84_disablecopyonread_adam_cblock_4_sepconv2d_bias_v^Read_84/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:c*
dtype0l
Identity_168IdentityRead_84/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:cc
Identity_169IdentityIdentity_168:output:0"/device:CPU:0*
T0*
_output_shapes
:c�
Read_85/DisableCopyOnReadDisableCopyOnRead9read_85_disablecopyonread_adam_cblock_4_batchnorm_gamma_v"/device:CPU:0*
_output_shapes
 �
Read_85/ReadVariableOpReadVariableOp9read_85_disablecopyonread_adam_cblock_4_batchnorm_gamma_v^Read_85/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:c*
dtype0l
Identity_170IdentityRead_85/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:cc
Identity_171IdentityIdentity_170:output:0"/device:CPU:0*
T0*
_output_shapes
:c�
Read_86/DisableCopyOnReadDisableCopyOnRead8read_86_disablecopyonread_adam_cblock_4_batchnorm_beta_v"/device:CPU:0*
_output_shapes
 �
Read_86/ReadVariableOpReadVariableOp8read_86_disablecopyonread_adam_cblock_4_batchnorm_beta_v^Read_86/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:c*
dtype0l
Identity_172IdentityRead_86/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:cc
Identity_173IdentityIdentity_172:output:0"/device:CPU:0*
T0*
_output_shapes
:c�
Read_87/DisableCopyOnReadDisableCopyOnRead/read_87_disablecopyonread_adam_dense_8_kernel_v"/device:CPU:0*
_output_shapes
 �
Read_87/ReadVariableOpReadVariableOp/read_87_disablecopyonread_adam_dense_8_kernel_v^Read_87/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:c*
dtype0p
Identity_174IdentityRead_87/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:cg
Identity_175IdentityIdentity_174:output:0"/device:CPU:0*
T0*
_output_shapes

:c�
Read_88/DisableCopyOnReadDisableCopyOnRead-read_88_disablecopyonread_adam_dense_8_bias_v"/device:CPU:0*
_output_shapes
 �
Read_88/ReadVariableOpReadVariableOp-read_88_disablecopyonread_adam_dense_8_bias_v^Read_88/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_176IdentityRead_88/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_177IdentityIdentity_176:output:0"/device:CPU:0*
T0*
_output_shapes
:�3
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:Z*
dtype0*�3
value�3B�3ZB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-1/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-1/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-3/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-3/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-5/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-5/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-7/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-7/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-8/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-8/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-8/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-1/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-1/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-3/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-3/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-5/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-5/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-7/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-7/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-1/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-1/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-3/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-3/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-5/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-5/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-7/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-7/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:Z*
dtype0*�
value�B�ZB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0Identity_23:output:0Identity_25:output:0Identity_27:output:0Identity_29:output:0Identity_31:output:0Identity_33:output:0Identity_35:output:0Identity_37:output:0Identity_39:output:0Identity_41:output:0Identity_43:output:0Identity_45:output:0Identity_47:output:0Identity_49:output:0Identity_51:output:0Identity_53:output:0Identity_55:output:0Identity_57:output:0Identity_59:output:0Identity_61:output:0Identity_63:output:0Identity_65:output:0Identity_67:output:0Identity_69:output:0Identity_71:output:0Identity_73:output:0Identity_75:output:0Identity_77:output:0Identity_79:output:0Identity_81:output:0Identity_83:output:0Identity_85:output:0Identity_87:output:0Identity_89:output:0Identity_91:output:0Identity_93:output:0Identity_95:output:0Identity_97:output:0Identity_99:output:0Identity_101:output:0Identity_103:output:0Identity_105:output:0Identity_107:output:0Identity_109:output:0Identity_111:output:0Identity_113:output:0Identity_115:output:0Identity_117:output:0Identity_119:output:0Identity_121:output:0Identity_123:output:0Identity_125:output:0Identity_127:output:0Identity_129:output:0Identity_131:output:0Identity_133:output:0Identity_135:output:0Identity_137:output:0Identity_139:output:0Identity_141:output:0Identity_143:output:0Identity_145:output:0Identity_147:output:0Identity_149:output:0Identity_151:output:0Identity_153:output:0Identity_155:output:0Identity_157:output:0Identity_159:output:0Identity_161:output:0Identity_163:output:0Identity_165:output:0Identity_167:output:0Identity_169:output:0Identity_171:output:0Identity_173:output:0Identity_175:output:0Identity_177:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *h
dtypes^
\2Z	�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 j
Identity_178Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: W
Identity_179IdentityIdentity_178:output:0^NoOp*
T0*
_output_shapes
: �%
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_10/DisableCopyOnRead^Read_10/ReadVariableOp^Read_11/DisableCopyOnRead^Read_11/ReadVariableOp^Read_12/DisableCopyOnRead^Read_12/ReadVariableOp^Read_13/DisableCopyOnRead^Read_13/ReadVariableOp^Read_14/DisableCopyOnRead^Read_14/ReadVariableOp^Read_15/DisableCopyOnRead^Read_15/ReadVariableOp^Read_16/DisableCopyOnRead^Read_16/ReadVariableOp^Read_17/DisableCopyOnRead^Read_17/ReadVariableOp^Read_18/DisableCopyOnRead^Read_18/ReadVariableOp^Read_19/DisableCopyOnRead^Read_19/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_20/DisableCopyOnRead^Read_20/ReadVariableOp^Read_21/DisableCopyOnRead^Read_21/ReadVariableOp^Read_22/DisableCopyOnRead^Read_22/ReadVariableOp^Read_23/DisableCopyOnRead^Read_23/ReadVariableOp^Read_24/DisableCopyOnRead^Read_24/ReadVariableOp^Read_25/DisableCopyOnRead^Read_25/ReadVariableOp^Read_26/DisableCopyOnRead^Read_26/ReadVariableOp^Read_27/DisableCopyOnRead^Read_27/ReadVariableOp^Read_28/DisableCopyOnRead^Read_28/ReadVariableOp^Read_29/DisableCopyOnRead^Read_29/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_30/DisableCopyOnRead^Read_30/ReadVariableOp^Read_31/DisableCopyOnRead^Read_31/ReadVariableOp^Read_32/DisableCopyOnRead^Read_32/ReadVariableOp^Read_33/DisableCopyOnRead^Read_33/ReadVariableOp^Read_34/DisableCopyOnRead^Read_34/ReadVariableOp^Read_35/DisableCopyOnRead^Read_35/ReadVariableOp^Read_36/DisableCopyOnRead^Read_36/ReadVariableOp^Read_37/DisableCopyOnRead^Read_37/ReadVariableOp^Read_38/DisableCopyOnRead^Read_38/ReadVariableOp^Read_39/DisableCopyOnRead^Read_39/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_40/DisableCopyOnRead^Read_40/ReadVariableOp^Read_41/DisableCopyOnRead^Read_41/ReadVariableOp^Read_42/DisableCopyOnRead^Read_42/ReadVariableOp^Read_43/DisableCopyOnRead^Read_43/ReadVariableOp^Read_44/DisableCopyOnRead^Read_44/ReadVariableOp^Read_45/DisableCopyOnRead^Read_45/ReadVariableOp^Read_46/DisableCopyOnRead^Read_46/ReadVariableOp^Read_47/DisableCopyOnRead^Read_47/ReadVariableOp^Read_48/DisableCopyOnRead^Read_48/ReadVariableOp^Read_49/DisableCopyOnRead^Read_49/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_50/DisableCopyOnRead^Read_50/ReadVariableOp^Read_51/DisableCopyOnRead^Read_51/ReadVariableOp^Read_52/DisableCopyOnRead^Read_52/ReadVariableOp^Read_53/DisableCopyOnRead^Read_53/ReadVariableOp^Read_54/DisableCopyOnRead^Read_54/ReadVariableOp^Read_55/DisableCopyOnRead^Read_55/ReadVariableOp^Read_56/DisableCopyOnRead^Read_56/ReadVariableOp^Read_57/DisableCopyOnRead^Read_57/ReadVariableOp^Read_58/DisableCopyOnRead^Read_58/ReadVariableOp^Read_59/DisableCopyOnRead^Read_59/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_60/DisableCopyOnRead^Read_60/ReadVariableOp^Read_61/DisableCopyOnRead^Read_61/ReadVariableOp^Read_62/DisableCopyOnRead^Read_62/ReadVariableOp^Read_63/DisableCopyOnRead^Read_63/ReadVariableOp^Read_64/DisableCopyOnRead^Read_64/ReadVariableOp^Read_65/DisableCopyOnRead^Read_65/ReadVariableOp^Read_66/DisableCopyOnRead^Read_66/ReadVariableOp^Read_67/DisableCopyOnRead^Read_67/ReadVariableOp^Read_68/DisableCopyOnRead^Read_68/ReadVariableOp^Read_69/DisableCopyOnRead^Read_69/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_70/DisableCopyOnRead^Read_70/ReadVariableOp^Read_71/DisableCopyOnRead^Read_71/ReadVariableOp^Read_72/DisableCopyOnRead^Read_72/ReadVariableOp^Read_73/DisableCopyOnRead^Read_73/ReadVariableOp^Read_74/DisableCopyOnRead^Read_74/ReadVariableOp^Read_75/DisableCopyOnRead^Read_75/ReadVariableOp^Read_76/DisableCopyOnRead^Read_76/ReadVariableOp^Read_77/DisableCopyOnRead^Read_77/ReadVariableOp^Read_78/DisableCopyOnRead^Read_78/ReadVariableOp^Read_79/DisableCopyOnRead^Read_79/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp^Read_80/DisableCopyOnRead^Read_80/ReadVariableOp^Read_81/DisableCopyOnRead^Read_81/ReadVariableOp^Read_82/DisableCopyOnRead^Read_82/ReadVariableOp^Read_83/DisableCopyOnRead^Read_83/ReadVariableOp^Read_84/DisableCopyOnRead^Read_84/ReadVariableOp^Read_85/DisableCopyOnRead^Read_85/ReadVariableOp^Read_86/DisableCopyOnRead^Read_86/ReadVariableOp^Read_87/DisableCopyOnRead^Read_87/ReadVariableOp^Read_88/DisableCopyOnRead^Read_88/ReadVariableOp^Read_9/DisableCopyOnRead^Read_9/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "%
identity_179Identity_179:output:0*�
_input_shapes�
�: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints20
Read/DisableCopyOnReadRead/DisableCopyOnRead2*
Read/ReadVariableOpRead/ReadVariableOp24
Read_1/DisableCopyOnReadRead_1/DisableCopyOnRead2.
Read_1/ReadVariableOpRead_1/ReadVariableOp26
Read_10/DisableCopyOnReadRead_10/DisableCopyOnRead20
Read_10/ReadVariableOpRead_10/ReadVariableOp26
Read_11/DisableCopyOnReadRead_11/DisableCopyOnRead20
Read_11/ReadVariableOpRead_11/ReadVariableOp26
Read_12/DisableCopyOnReadRead_12/DisableCopyOnRead20
Read_12/ReadVariableOpRead_12/ReadVariableOp26
Read_13/DisableCopyOnReadRead_13/DisableCopyOnRead20
Read_13/ReadVariableOpRead_13/ReadVariableOp26
Read_14/DisableCopyOnReadRead_14/DisableCopyOnRead20
Read_14/ReadVariableOpRead_14/ReadVariableOp26
Read_15/DisableCopyOnReadRead_15/DisableCopyOnRead20
Read_15/ReadVariableOpRead_15/ReadVariableOp26
Read_16/DisableCopyOnReadRead_16/DisableCopyOnRead20
Read_16/ReadVariableOpRead_16/ReadVariableOp26
Read_17/DisableCopyOnReadRead_17/DisableCopyOnRead20
Read_17/ReadVariableOpRead_17/ReadVariableOp26
Read_18/DisableCopyOnReadRead_18/DisableCopyOnRead20
Read_18/ReadVariableOpRead_18/ReadVariableOp26
Read_19/DisableCopyOnReadRead_19/DisableCopyOnRead20
Read_19/ReadVariableOpRead_19/ReadVariableOp24
Read_2/DisableCopyOnReadRead_2/DisableCopyOnRead2.
Read_2/ReadVariableOpRead_2/ReadVariableOp26
Read_20/DisableCopyOnReadRead_20/DisableCopyOnRead20
Read_20/ReadVariableOpRead_20/ReadVariableOp26
Read_21/DisableCopyOnReadRead_21/DisableCopyOnRead20
Read_21/ReadVariableOpRead_21/ReadVariableOp26
Read_22/DisableCopyOnReadRead_22/DisableCopyOnRead20
Read_22/ReadVariableOpRead_22/ReadVariableOp26
Read_23/DisableCopyOnReadRead_23/DisableCopyOnRead20
Read_23/ReadVariableOpRead_23/ReadVariableOp26
Read_24/DisableCopyOnReadRead_24/DisableCopyOnRead20
Read_24/ReadVariableOpRead_24/ReadVariableOp26
Read_25/DisableCopyOnReadRead_25/DisableCopyOnRead20
Read_25/ReadVariableOpRead_25/ReadVariableOp26
Read_26/DisableCopyOnReadRead_26/DisableCopyOnRead20
Read_26/ReadVariableOpRead_26/ReadVariableOp26
Read_27/DisableCopyOnReadRead_27/DisableCopyOnRead20
Read_27/ReadVariableOpRead_27/ReadVariableOp26
Read_28/DisableCopyOnReadRead_28/DisableCopyOnRead20
Read_28/ReadVariableOpRead_28/ReadVariableOp26
Read_29/DisableCopyOnReadRead_29/DisableCopyOnRead20
Read_29/ReadVariableOpRead_29/ReadVariableOp24
Read_3/DisableCopyOnReadRead_3/DisableCopyOnRead2.
Read_3/ReadVariableOpRead_3/ReadVariableOp26
Read_30/DisableCopyOnReadRead_30/DisableCopyOnRead20
Read_30/ReadVariableOpRead_30/ReadVariableOp26
Read_31/DisableCopyOnReadRead_31/DisableCopyOnRead20
Read_31/ReadVariableOpRead_31/ReadVariableOp26
Read_32/DisableCopyOnReadRead_32/DisableCopyOnRead20
Read_32/ReadVariableOpRead_32/ReadVariableOp26
Read_33/DisableCopyOnReadRead_33/DisableCopyOnRead20
Read_33/ReadVariableOpRead_33/ReadVariableOp26
Read_34/DisableCopyOnReadRead_34/DisableCopyOnRead20
Read_34/ReadVariableOpRead_34/ReadVariableOp26
Read_35/DisableCopyOnReadRead_35/DisableCopyOnRead20
Read_35/ReadVariableOpRead_35/ReadVariableOp26
Read_36/DisableCopyOnReadRead_36/DisableCopyOnRead20
Read_36/ReadVariableOpRead_36/ReadVariableOp26
Read_37/DisableCopyOnReadRead_37/DisableCopyOnRead20
Read_37/ReadVariableOpRead_37/ReadVariableOp26
Read_38/DisableCopyOnReadRead_38/DisableCopyOnRead20
Read_38/ReadVariableOpRead_38/ReadVariableOp26
Read_39/DisableCopyOnReadRead_39/DisableCopyOnRead20
Read_39/ReadVariableOpRead_39/ReadVariableOp24
Read_4/DisableCopyOnReadRead_4/DisableCopyOnRead2.
Read_4/ReadVariableOpRead_4/ReadVariableOp26
Read_40/DisableCopyOnReadRead_40/DisableCopyOnRead20
Read_40/ReadVariableOpRead_40/ReadVariableOp26
Read_41/DisableCopyOnReadRead_41/DisableCopyOnRead20
Read_41/ReadVariableOpRead_41/ReadVariableOp26
Read_42/DisableCopyOnReadRead_42/DisableCopyOnRead20
Read_42/ReadVariableOpRead_42/ReadVariableOp26
Read_43/DisableCopyOnReadRead_43/DisableCopyOnRead20
Read_43/ReadVariableOpRead_43/ReadVariableOp26
Read_44/DisableCopyOnReadRead_44/DisableCopyOnRead20
Read_44/ReadVariableOpRead_44/ReadVariableOp26
Read_45/DisableCopyOnReadRead_45/DisableCopyOnRead20
Read_45/ReadVariableOpRead_45/ReadVariableOp26
Read_46/DisableCopyOnReadRead_46/DisableCopyOnRead20
Read_46/ReadVariableOpRead_46/ReadVariableOp26
Read_47/DisableCopyOnReadRead_47/DisableCopyOnRead20
Read_47/ReadVariableOpRead_47/ReadVariableOp26
Read_48/DisableCopyOnReadRead_48/DisableCopyOnRead20
Read_48/ReadVariableOpRead_48/ReadVariableOp26
Read_49/DisableCopyOnReadRead_49/DisableCopyOnRead20
Read_49/ReadVariableOpRead_49/ReadVariableOp24
Read_5/DisableCopyOnReadRead_5/DisableCopyOnRead2.
Read_5/ReadVariableOpRead_5/ReadVariableOp26
Read_50/DisableCopyOnReadRead_50/DisableCopyOnRead20
Read_50/ReadVariableOpRead_50/ReadVariableOp26
Read_51/DisableCopyOnReadRead_51/DisableCopyOnRead20
Read_51/ReadVariableOpRead_51/ReadVariableOp26
Read_52/DisableCopyOnReadRead_52/DisableCopyOnRead20
Read_52/ReadVariableOpRead_52/ReadVariableOp26
Read_53/DisableCopyOnReadRead_53/DisableCopyOnRead20
Read_53/ReadVariableOpRead_53/ReadVariableOp26
Read_54/DisableCopyOnReadRead_54/DisableCopyOnRead20
Read_54/ReadVariableOpRead_54/ReadVariableOp26
Read_55/DisableCopyOnReadRead_55/DisableCopyOnRead20
Read_55/ReadVariableOpRead_55/ReadVariableOp26
Read_56/DisableCopyOnReadRead_56/DisableCopyOnRead20
Read_56/ReadVariableOpRead_56/ReadVariableOp26
Read_57/DisableCopyOnReadRead_57/DisableCopyOnRead20
Read_57/ReadVariableOpRead_57/ReadVariableOp26
Read_58/DisableCopyOnReadRead_58/DisableCopyOnRead20
Read_58/ReadVariableOpRead_58/ReadVariableOp26
Read_59/DisableCopyOnReadRead_59/DisableCopyOnRead20
Read_59/ReadVariableOpRead_59/ReadVariableOp24
Read_6/DisableCopyOnReadRead_6/DisableCopyOnRead2.
Read_6/ReadVariableOpRead_6/ReadVariableOp26
Read_60/DisableCopyOnReadRead_60/DisableCopyOnRead20
Read_60/ReadVariableOpRead_60/ReadVariableOp26
Read_61/DisableCopyOnReadRead_61/DisableCopyOnRead20
Read_61/ReadVariableOpRead_61/ReadVariableOp26
Read_62/DisableCopyOnReadRead_62/DisableCopyOnRead20
Read_62/ReadVariableOpRead_62/ReadVariableOp26
Read_63/DisableCopyOnReadRead_63/DisableCopyOnRead20
Read_63/ReadVariableOpRead_63/ReadVariableOp26
Read_64/DisableCopyOnReadRead_64/DisableCopyOnRead20
Read_64/ReadVariableOpRead_64/ReadVariableOp26
Read_65/DisableCopyOnReadRead_65/DisableCopyOnRead20
Read_65/ReadVariableOpRead_65/ReadVariableOp26
Read_66/DisableCopyOnReadRead_66/DisableCopyOnRead20
Read_66/ReadVariableOpRead_66/ReadVariableOp26
Read_67/DisableCopyOnReadRead_67/DisableCopyOnRead20
Read_67/ReadVariableOpRead_67/ReadVariableOp26
Read_68/DisableCopyOnReadRead_68/DisableCopyOnRead20
Read_68/ReadVariableOpRead_68/ReadVariableOp26
Read_69/DisableCopyOnReadRead_69/DisableCopyOnRead20
Read_69/ReadVariableOpRead_69/ReadVariableOp24
Read_7/DisableCopyOnReadRead_7/DisableCopyOnRead2.
Read_7/ReadVariableOpRead_7/ReadVariableOp26
Read_70/DisableCopyOnReadRead_70/DisableCopyOnRead20
Read_70/ReadVariableOpRead_70/ReadVariableOp26
Read_71/DisableCopyOnReadRead_71/DisableCopyOnRead20
Read_71/ReadVariableOpRead_71/ReadVariableOp26
Read_72/DisableCopyOnReadRead_72/DisableCopyOnRead20
Read_72/ReadVariableOpRead_72/ReadVariableOp26
Read_73/DisableCopyOnReadRead_73/DisableCopyOnRead20
Read_73/ReadVariableOpRead_73/ReadVariableOp26
Read_74/DisableCopyOnReadRead_74/DisableCopyOnRead20
Read_74/ReadVariableOpRead_74/ReadVariableOp26
Read_75/DisableCopyOnReadRead_75/DisableCopyOnRead20
Read_75/ReadVariableOpRead_75/ReadVariableOp26
Read_76/DisableCopyOnReadRead_76/DisableCopyOnRead20
Read_76/ReadVariableOpRead_76/ReadVariableOp26
Read_77/DisableCopyOnReadRead_77/DisableCopyOnRead20
Read_77/ReadVariableOpRead_77/ReadVariableOp26
Read_78/DisableCopyOnReadRead_78/DisableCopyOnRead20
Read_78/ReadVariableOpRead_78/ReadVariableOp26
Read_79/DisableCopyOnReadRead_79/DisableCopyOnRead20
Read_79/ReadVariableOpRead_79/ReadVariableOp24
Read_8/DisableCopyOnReadRead_8/DisableCopyOnRead2.
Read_8/ReadVariableOpRead_8/ReadVariableOp26
Read_80/DisableCopyOnReadRead_80/DisableCopyOnRead20
Read_80/ReadVariableOpRead_80/ReadVariableOp26
Read_81/DisableCopyOnReadRead_81/DisableCopyOnRead20
Read_81/ReadVariableOpRead_81/ReadVariableOp26
Read_82/DisableCopyOnReadRead_82/DisableCopyOnRead20
Read_82/ReadVariableOpRead_82/ReadVariableOp26
Read_83/DisableCopyOnReadRead_83/DisableCopyOnRead20
Read_83/ReadVariableOpRead_83/ReadVariableOp26
Read_84/DisableCopyOnReadRead_84/DisableCopyOnRead20
Read_84/ReadVariableOpRead_84/ReadVariableOp26
Read_85/DisableCopyOnReadRead_85/DisableCopyOnRead20
Read_85/ReadVariableOpRead_85/ReadVariableOp26
Read_86/DisableCopyOnReadRead_86/DisableCopyOnRead20
Read_86/ReadVariableOpRead_86/ReadVariableOp26
Read_87/DisableCopyOnReadRead_87/DisableCopyOnRead20
Read_87/ReadVariableOpRead_87/ReadVariableOp26
Read_88/DisableCopyOnReadRead_88/DisableCopyOnRead20
Read_88/ReadVariableOpRead_88/ReadVariableOp24
Read_9/DisableCopyOnReadRead_9/DisableCopyOnRead2.
Read_9/ReadVariableOpRead_9/ReadVariableOp:Z

_output_shapes
: :C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
�
O__inference_CBlock_1_SepConv2D_layer_call_and_return_conditional_losses_1690315

inputsB
(separable_conv2d_readvariableop_resource:D
*separable_conv2d_readvariableop_1_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�separable_conv2d/ReadVariableOp�!separable_conv2d/ReadVariableOp_1�
separable_conv2d/ReadVariableOpReadVariableOp(separable_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
!separable_conv2d/ReadVariableOp_1ReadVariableOp*separable_conv2d_readvariableop_1_resource*&
_output_shapes
:*
dtype0o
separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            o
separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      �
separable_conv2d/depthwiseDepthwiseConv2dNativeinputs'separable_conv2d/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������*
paddingSAME*
strides
�
separable_conv2dConv2D#separable_conv2d/depthwise:output:0)separable_conv2d/ReadVariableOp_1:value:0*
T0*A
_output_shapes/
-:+���������������������������*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
BiasAddBiasAddseparable_conv2d:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������y
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+����������������������������
NoOpNoOp^BiasAdd/ReadVariableOp ^separable_conv2d/ReadVariableOp"^separable_conv2d/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:+���������������������������: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2F
!separable_conv2d/ReadVariableOp_1!separable_conv2d/ReadVariableOp_12B
separable_conv2d/ReadVariableOpseparable_conv2d/ReadVariableOp:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
f
J__inference_CBlock_1_ReLu_layer_call_and_return_conditional_losses_1689149

inputs
identityP
Relu6Relu6inputs*
T0*/
_output_shapes
:���������c
IdentityIdentityRelu6:activations:0*
T0*/
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�	
�
4__inference_CBlock_4_BatchNorm_layer_call_fn_1690622

inputs
unknown:c
	unknown_0:c
	unknown_1:c
	unknown_2:c
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������c*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *X
fSRQ
O__inference_CBlock_4_BatchNorm_layer_call_and_return_conditional_losses_1689050�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������c`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������c: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������c
 
_user_specified_nameinputs
��
�>
#__inference__traced_restore_1691546
file_prefix7
assignvariableop_conv0_kernel:+
assignvariableop_1_conv0_bias:P
6assignvariableop_2_cblock_1_sepconv2d_depthwise_kernel:P
6assignvariableop_3_cblock_1_sepconv2d_pointwise_kernel:8
*assignvariableop_4_cblock_1_sepconv2d_bias:9
+assignvariableop_5_cblock_1_batchnorm_gamma:8
*assignvariableop_6_cblock_1_batchnorm_beta:?
1assignvariableop_7_cblock_1_batchnorm_moving_mean:C
5assignvariableop_8_cblock_1_batchnorm_moving_variance:P
6assignvariableop_9_cblock_2_sepconv2d_depthwise_kernel:Q
7assignvariableop_10_cblock_2_sepconv2d_pointwise_kernel:19
+assignvariableop_11_cblock_2_sepconv2d_bias:1:
,assignvariableop_12_cblock_2_batchnorm_gamma:19
+assignvariableop_13_cblock_2_batchnorm_beta:1@
2assignvariableop_14_cblock_2_batchnorm_moving_mean:1D
6assignvariableop_15_cblock_2_batchnorm_moving_variance:1Q
7assignvariableop_16_cblock_3_sepconv2d_depthwise_kernel:1Q
7assignvariableop_17_cblock_3_sepconv2d_pointwise_kernel:1J9
+assignvariableop_18_cblock_3_sepconv2d_bias:J:
,assignvariableop_19_cblock_3_batchnorm_gamma:J9
+assignvariableop_20_cblock_3_batchnorm_beta:J@
2assignvariableop_21_cblock_3_batchnorm_moving_mean:JD
6assignvariableop_22_cblock_3_batchnorm_moving_variance:JQ
7assignvariableop_23_cblock_4_sepconv2d_depthwise_kernel:JQ
7assignvariableop_24_cblock_4_sepconv2d_pointwise_kernel:Jc9
+assignvariableop_25_cblock_4_sepconv2d_bias:c:
,assignvariableop_26_cblock_4_batchnorm_gamma:c9
+assignvariableop_27_cblock_4_batchnorm_beta:c@
2assignvariableop_28_cblock_4_batchnorm_moving_mean:cD
6assignvariableop_29_cblock_4_batchnorm_moving_variance:c4
"assignvariableop_30_dense_8_kernel:c.
 assignvariableop_31_dense_8_bias:'
assignvariableop_32_adam_iter:	 )
assignvariableop_33_adam_beta_1: )
assignvariableop_34_adam_beta_2: (
assignvariableop_35_adam_decay: 0
&assignvariableop_36_adam_learning_rate: %
assignvariableop_37_total_1: %
assignvariableop_38_count_1: #
assignvariableop_39_total: #
assignvariableop_40_count: A
'assignvariableop_41_adam_conv0_kernel_m:3
%assignvariableop_42_adam_conv0_bias_m:X
>assignvariableop_43_adam_cblock_1_sepconv2d_depthwise_kernel_m:X
>assignvariableop_44_adam_cblock_1_sepconv2d_pointwise_kernel_m:@
2assignvariableop_45_adam_cblock_1_sepconv2d_bias_m:A
3assignvariableop_46_adam_cblock_1_batchnorm_gamma_m:@
2assignvariableop_47_adam_cblock_1_batchnorm_beta_m:X
>assignvariableop_48_adam_cblock_2_sepconv2d_depthwise_kernel_m:X
>assignvariableop_49_adam_cblock_2_sepconv2d_pointwise_kernel_m:1@
2assignvariableop_50_adam_cblock_2_sepconv2d_bias_m:1A
3assignvariableop_51_adam_cblock_2_batchnorm_gamma_m:1@
2assignvariableop_52_adam_cblock_2_batchnorm_beta_m:1X
>assignvariableop_53_adam_cblock_3_sepconv2d_depthwise_kernel_m:1X
>assignvariableop_54_adam_cblock_3_sepconv2d_pointwise_kernel_m:1J@
2assignvariableop_55_adam_cblock_3_sepconv2d_bias_m:JA
3assignvariableop_56_adam_cblock_3_batchnorm_gamma_m:J@
2assignvariableop_57_adam_cblock_3_batchnorm_beta_m:JX
>assignvariableop_58_adam_cblock_4_sepconv2d_depthwise_kernel_m:JX
>assignvariableop_59_adam_cblock_4_sepconv2d_pointwise_kernel_m:Jc@
2assignvariableop_60_adam_cblock_4_sepconv2d_bias_m:cA
3assignvariableop_61_adam_cblock_4_batchnorm_gamma_m:c@
2assignvariableop_62_adam_cblock_4_batchnorm_beta_m:c;
)assignvariableop_63_adam_dense_8_kernel_m:c5
'assignvariableop_64_adam_dense_8_bias_m:A
'assignvariableop_65_adam_conv0_kernel_v:3
%assignvariableop_66_adam_conv0_bias_v:X
>assignvariableop_67_adam_cblock_1_sepconv2d_depthwise_kernel_v:X
>assignvariableop_68_adam_cblock_1_sepconv2d_pointwise_kernel_v:@
2assignvariableop_69_adam_cblock_1_sepconv2d_bias_v:A
3assignvariableop_70_adam_cblock_1_batchnorm_gamma_v:@
2assignvariableop_71_adam_cblock_1_batchnorm_beta_v:X
>assignvariableop_72_adam_cblock_2_sepconv2d_depthwise_kernel_v:X
>assignvariableop_73_adam_cblock_2_sepconv2d_pointwise_kernel_v:1@
2assignvariableop_74_adam_cblock_2_sepconv2d_bias_v:1A
3assignvariableop_75_adam_cblock_2_batchnorm_gamma_v:1@
2assignvariableop_76_adam_cblock_2_batchnorm_beta_v:1X
>assignvariableop_77_adam_cblock_3_sepconv2d_depthwise_kernel_v:1X
>assignvariableop_78_adam_cblock_3_sepconv2d_pointwise_kernel_v:1J@
2assignvariableop_79_adam_cblock_3_sepconv2d_bias_v:JA
3assignvariableop_80_adam_cblock_3_batchnorm_gamma_v:J@
2assignvariableop_81_adam_cblock_3_batchnorm_beta_v:JX
>assignvariableop_82_adam_cblock_4_sepconv2d_depthwise_kernel_v:JX
>assignvariableop_83_adam_cblock_4_sepconv2d_pointwise_kernel_v:Jc@
2assignvariableop_84_adam_cblock_4_sepconv2d_bias_v:cA
3assignvariableop_85_adam_cblock_4_batchnorm_gamma_v:c@
2assignvariableop_86_adam_cblock_4_batchnorm_beta_v:c;
)assignvariableop_87_adam_dense_8_kernel_v:c5
'assignvariableop_88_adam_dense_8_bias_v:
identity_90��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_36�AssignVariableOp_37�AssignVariableOp_38�AssignVariableOp_39�AssignVariableOp_4�AssignVariableOp_40�AssignVariableOp_41�AssignVariableOp_42�AssignVariableOp_43�AssignVariableOp_44�AssignVariableOp_45�AssignVariableOp_46�AssignVariableOp_47�AssignVariableOp_48�AssignVariableOp_49�AssignVariableOp_5�AssignVariableOp_50�AssignVariableOp_51�AssignVariableOp_52�AssignVariableOp_53�AssignVariableOp_54�AssignVariableOp_55�AssignVariableOp_56�AssignVariableOp_57�AssignVariableOp_58�AssignVariableOp_59�AssignVariableOp_6�AssignVariableOp_60�AssignVariableOp_61�AssignVariableOp_62�AssignVariableOp_63�AssignVariableOp_64�AssignVariableOp_65�AssignVariableOp_66�AssignVariableOp_67�AssignVariableOp_68�AssignVariableOp_69�AssignVariableOp_7�AssignVariableOp_70�AssignVariableOp_71�AssignVariableOp_72�AssignVariableOp_73�AssignVariableOp_74�AssignVariableOp_75�AssignVariableOp_76�AssignVariableOp_77�AssignVariableOp_78�AssignVariableOp_79�AssignVariableOp_8�AssignVariableOp_80�AssignVariableOp_81�AssignVariableOp_82�AssignVariableOp_83�AssignVariableOp_84�AssignVariableOp_85�AssignVariableOp_86�AssignVariableOp_87�AssignVariableOp_88�AssignVariableOp_9�3
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:Z*
dtype0*�3
value�3B�3ZB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-1/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-1/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-3/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-3/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-5/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-5/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-7/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-7/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-8/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-8/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-8/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-1/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-1/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-3/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-3/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-5/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-5/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-7/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-7/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-1/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-1/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-3/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-3/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-5/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-5/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-7/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-7/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:Z*
dtype0*�
value�B�ZB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*h
dtypes^
\2Z	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOpassignvariableop_conv0_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOpassignvariableop_1_conv0_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp6assignvariableop_2_cblock_1_sepconv2d_depthwise_kernelIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp6assignvariableop_3_cblock_1_sepconv2d_pointwise_kernelIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp*assignvariableop_4_cblock_1_sepconv2d_biasIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp+assignvariableop_5_cblock_1_batchnorm_gammaIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp*assignvariableop_6_cblock_1_batchnorm_betaIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp1assignvariableop_7_cblock_1_batchnorm_moving_meanIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp5assignvariableop_8_cblock_1_batchnorm_moving_varianceIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp6assignvariableop_9_cblock_2_sepconv2d_depthwise_kernelIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOp7assignvariableop_10_cblock_2_sepconv2d_pointwise_kernelIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOp+assignvariableop_11_cblock_2_sepconv2d_biasIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOp,assignvariableop_12_cblock_2_batchnorm_gammaIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOp+assignvariableop_13_cblock_2_batchnorm_betaIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOp2assignvariableop_14_cblock_2_batchnorm_moving_meanIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOp6assignvariableop_15_cblock_2_batchnorm_moving_varianceIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOp7assignvariableop_16_cblock_3_sepconv2d_depthwise_kernelIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOp7assignvariableop_17_cblock_3_sepconv2d_pointwise_kernelIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOp+assignvariableop_18_cblock_3_sepconv2d_biasIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOp,assignvariableop_19_cblock_3_batchnorm_gammaIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOp+assignvariableop_20_cblock_3_batchnorm_betaIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOp2assignvariableop_21_cblock_3_batchnorm_moving_meanIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOp6assignvariableop_22_cblock_3_batchnorm_moving_varianceIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOp7assignvariableop_23_cblock_4_sepconv2d_depthwise_kernelIdentity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_24AssignVariableOp7assignvariableop_24_cblock_4_sepconv2d_pointwise_kernelIdentity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_25AssignVariableOp+assignvariableop_25_cblock_4_sepconv2d_biasIdentity_25:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_26AssignVariableOp,assignvariableop_26_cblock_4_batchnorm_gammaIdentity_26:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_27AssignVariableOp+assignvariableop_27_cblock_4_batchnorm_betaIdentity_27:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_28AssignVariableOp2assignvariableop_28_cblock_4_batchnorm_moving_meanIdentity_28:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_29AssignVariableOp6assignvariableop_29_cblock_4_batchnorm_moving_varianceIdentity_29:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_30AssignVariableOp"assignvariableop_30_dense_8_kernelIdentity_30:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_31AssignVariableOp assignvariableop_31_dense_8_biasIdentity_31:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_32AssignVariableOpassignvariableop_32_adam_iterIdentity_32:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_33AssignVariableOpassignvariableop_33_adam_beta_1Identity_33:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_34AssignVariableOpassignvariableop_34_adam_beta_2Identity_34:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_35AssignVariableOpassignvariableop_35_adam_decayIdentity_35:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_36AssignVariableOp&assignvariableop_36_adam_learning_rateIdentity_36:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_37AssignVariableOpassignvariableop_37_total_1Identity_37:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_38AssignVariableOpassignvariableop_38_count_1Identity_38:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_39AssignVariableOpassignvariableop_39_totalIdentity_39:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_40AssignVariableOpassignvariableop_40_countIdentity_40:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_41AssignVariableOp'assignvariableop_41_adam_conv0_kernel_mIdentity_41:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_42AssignVariableOp%assignvariableop_42_adam_conv0_bias_mIdentity_42:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_43AssignVariableOp>assignvariableop_43_adam_cblock_1_sepconv2d_depthwise_kernel_mIdentity_43:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_44AssignVariableOp>assignvariableop_44_adam_cblock_1_sepconv2d_pointwise_kernel_mIdentity_44:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_45AssignVariableOp2assignvariableop_45_adam_cblock_1_sepconv2d_bias_mIdentity_45:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_46AssignVariableOp3assignvariableop_46_adam_cblock_1_batchnorm_gamma_mIdentity_46:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_47AssignVariableOp2assignvariableop_47_adam_cblock_1_batchnorm_beta_mIdentity_47:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_48AssignVariableOp>assignvariableop_48_adam_cblock_2_sepconv2d_depthwise_kernel_mIdentity_48:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_49AssignVariableOp>assignvariableop_49_adam_cblock_2_sepconv2d_pointwise_kernel_mIdentity_49:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_50AssignVariableOp2assignvariableop_50_adam_cblock_2_sepconv2d_bias_mIdentity_50:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_51AssignVariableOp3assignvariableop_51_adam_cblock_2_batchnorm_gamma_mIdentity_51:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_52AssignVariableOp2assignvariableop_52_adam_cblock_2_batchnorm_beta_mIdentity_52:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_53AssignVariableOp>assignvariableop_53_adam_cblock_3_sepconv2d_depthwise_kernel_mIdentity_53:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_54AssignVariableOp>assignvariableop_54_adam_cblock_3_sepconv2d_pointwise_kernel_mIdentity_54:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_55AssignVariableOp2assignvariableop_55_adam_cblock_3_sepconv2d_bias_mIdentity_55:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_56AssignVariableOp3assignvariableop_56_adam_cblock_3_batchnorm_gamma_mIdentity_56:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_57AssignVariableOp2assignvariableop_57_adam_cblock_3_batchnorm_beta_mIdentity_57:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_58AssignVariableOp>assignvariableop_58_adam_cblock_4_sepconv2d_depthwise_kernel_mIdentity_58:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_59AssignVariableOp>assignvariableop_59_adam_cblock_4_sepconv2d_pointwise_kernel_mIdentity_59:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_60AssignVariableOp2assignvariableop_60_adam_cblock_4_sepconv2d_bias_mIdentity_60:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_61AssignVariableOp3assignvariableop_61_adam_cblock_4_batchnorm_gamma_mIdentity_61:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_62AssignVariableOp2assignvariableop_62_adam_cblock_4_batchnorm_beta_mIdentity_62:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_63AssignVariableOp)assignvariableop_63_adam_dense_8_kernel_mIdentity_63:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_64AssignVariableOp'assignvariableop_64_adam_dense_8_bias_mIdentity_64:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_65AssignVariableOp'assignvariableop_65_adam_conv0_kernel_vIdentity_65:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_66AssignVariableOp%assignvariableop_66_adam_conv0_bias_vIdentity_66:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_67AssignVariableOp>assignvariableop_67_adam_cblock_1_sepconv2d_depthwise_kernel_vIdentity_67:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_68AssignVariableOp>assignvariableop_68_adam_cblock_1_sepconv2d_pointwise_kernel_vIdentity_68:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_69AssignVariableOp2assignvariableop_69_adam_cblock_1_sepconv2d_bias_vIdentity_69:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_70AssignVariableOp3assignvariableop_70_adam_cblock_1_batchnorm_gamma_vIdentity_70:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_71AssignVariableOp2assignvariableop_71_adam_cblock_1_batchnorm_beta_vIdentity_71:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_72AssignVariableOp>assignvariableop_72_adam_cblock_2_sepconv2d_depthwise_kernel_vIdentity_72:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_73AssignVariableOp>assignvariableop_73_adam_cblock_2_sepconv2d_pointwise_kernel_vIdentity_73:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_74AssignVariableOp2assignvariableop_74_adam_cblock_2_sepconv2d_bias_vIdentity_74:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_75AssignVariableOp3assignvariableop_75_adam_cblock_2_batchnorm_gamma_vIdentity_75:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_76AssignVariableOp2assignvariableop_76_adam_cblock_2_batchnorm_beta_vIdentity_76:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_77AssignVariableOp>assignvariableop_77_adam_cblock_3_sepconv2d_depthwise_kernel_vIdentity_77:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_78AssignVariableOp>assignvariableop_78_adam_cblock_3_sepconv2d_pointwise_kernel_vIdentity_78:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_79AssignVariableOp2assignvariableop_79_adam_cblock_3_sepconv2d_bias_vIdentity_79:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_80AssignVariableOp3assignvariableop_80_adam_cblock_3_batchnorm_gamma_vIdentity_80:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_81AssignVariableOp2assignvariableop_81_adam_cblock_3_batchnorm_beta_vIdentity_81:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_82AssignVariableOp>assignvariableop_82_adam_cblock_4_sepconv2d_depthwise_kernel_vIdentity_82:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_83IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_83AssignVariableOp>assignvariableop_83_adam_cblock_4_sepconv2d_pointwise_kernel_vIdentity_83:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_84IdentityRestoreV2:tensors:84"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_84AssignVariableOp2assignvariableop_84_adam_cblock_4_sepconv2d_bias_vIdentity_84:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_85IdentityRestoreV2:tensors:85"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_85AssignVariableOp3assignvariableop_85_adam_cblock_4_batchnorm_gamma_vIdentity_85:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_86IdentityRestoreV2:tensors:86"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_86AssignVariableOp2assignvariableop_86_adam_cblock_4_batchnorm_beta_vIdentity_86:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_87IdentityRestoreV2:tensors:87"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_87AssignVariableOp)assignvariableop_87_adam_dense_8_kernel_vIdentity_87:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_88IdentityRestoreV2:tensors:88"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_88AssignVariableOp'assignvariableop_88_adam_dense_8_bias_vIdentity_88:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �
Identity_89Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_90IdentityIdentity_89:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_90Identity_90:output:0*�
_input_shapes�
�: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2*
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
AssignVariableOp_1AssignVariableOp_12*
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
AssignVariableOp_2AssignVariableOp_22*
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
AssignVariableOp_3AssignVariableOp_32*
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
AssignVariableOp_4AssignVariableOp_42*
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
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742*
AssignVariableOp_75AssignVariableOp_752*
AssignVariableOp_76AssignVariableOp_762*
AssignVariableOp_77AssignVariableOp_772*
AssignVariableOp_78AssignVariableOp_782*
AssignVariableOp_79AssignVariableOp_792(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_80AssignVariableOp_802*
AssignVariableOp_81AssignVariableOp_812*
AssignVariableOp_82AssignVariableOp_822*
AssignVariableOp_83AssignVariableOp_832*
AssignVariableOp_84AssignVariableOp_842*
AssignVariableOp_85AssignVariableOp_852*
AssignVariableOp_86AssignVariableOp_862*
AssignVariableOp_87AssignVariableOp_872*
AssignVariableOp_88AssignVariableOp_882(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92$
AssignVariableOpAssignVariableOp:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
�
F__inference_jojo_n8_r72x88_a0.7_b4_g2.2_strides2_layer_call_fn_1689475
input_layer!
unknown:
	unknown_0:#
	unknown_1:#
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:#
	unknown_8:#
	unknown_9:1

unknown_10:1

unknown_11:1

unknown_12:1

unknown_13:1

unknown_14:1$

unknown_15:1$

unknown_16:1J

unknown_17:J

unknown_18:J

unknown_19:J

unknown_20:J

unknown_21:J$

unknown_22:J$

unknown_23:Jc

unknown_24:c

unknown_25:c

unknown_26:c

unknown_27:c

unknown_28:c

unknown_29:c

unknown_30:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_layerunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30*,
Tin%
#2!*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*:
_read_only_resource_inputs

 *0
config_proto 

CPU

GPU2*0J 8� *j
feRc
a__inference_jojo_n8_r72x88_a0.7_b4_g2.2_strides2_layer_call_and_return_conditional_losses_1689408o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*n
_input_shapes]
[:���������HX: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
/
_output_shapes
:���������HX
%
_user_specified_nameinput_layer
��
�
a__inference_jojo_n8_r72x88_a0.7_b4_g2.2_strides2_layer_call_and_return_conditional_losses_1690270

inputs>
$conv0_conv2d_readvariableop_resource:3
%conv0_biasadd_readvariableop_resource:U
;cblock_1_sepconv2d_separable_conv2d_readvariableop_resource:W
=cblock_1_sepconv2d_separable_conv2d_readvariableop_1_resource:@
2cblock_1_sepconv2d_biasadd_readvariableop_resource:8
*cblock_1_batchnorm_readvariableop_resource::
,cblock_1_batchnorm_readvariableop_1_resource:I
;cblock_1_batchnorm_fusedbatchnormv3_readvariableop_resource:K
=cblock_1_batchnorm_fusedbatchnormv3_readvariableop_1_resource:U
;cblock_2_sepconv2d_separable_conv2d_readvariableop_resource:W
=cblock_2_sepconv2d_separable_conv2d_readvariableop_1_resource:1@
2cblock_2_sepconv2d_biasadd_readvariableop_resource:18
*cblock_2_batchnorm_readvariableop_resource:1:
,cblock_2_batchnorm_readvariableop_1_resource:1I
;cblock_2_batchnorm_fusedbatchnormv3_readvariableop_resource:1K
=cblock_2_batchnorm_fusedbatchnormv3_readvariableop_1_resource:1U
;cblock_3_sepconv2d_separable_conv2d_readvariableop_resource:1W
=cblock_3_sepconv2d_separable_conv2d_readvariableop_1_resource:1J@
2cblock_3_sepconv2d_biasadd_readvariableop_resource:J8
*cblock_3_batchnorm_readvariableop_resource:J:
,cblock_3_batchnorm_readvariableop_1_resource:JI
;cblock_3_batchnorm_fusedbatchnormv3_readvariableop_resource:JK
=cblock_3_batchnorm_fusedbatchnormv3_readvariableop_1_resource:JU
;cblock_4_sepconv2d_separable_conv2d_readvariableop_resource:JW
=cblock_4_sepconv2d_separable_conv2d_readvariableop_1_resource:Jc@
2cblock_4_sepconv2d_biasadd_readvariableop_resource:c8
*cblock_4_batchnorm_readvariableop_resource:c:
,cblock_4_batchnorm_readvariableop_1_resource:cI
;cblock_4_batchnorm_fusedbatchnormv3_readvariableop_resource:cK
=cblock_4_batchnorm_fusedbatchnormv3_readvariableop_1_resource:c8
&dense_8_matmul_readvariableop_resource:c5
'dense_8_biasadd_readvariableop_resource:
identity��2CBlock_1_BatchNorm/FusedBatchNormV3/ReadVariableOp�4CBlock_1_BatchNorm/FusedBatchNormV3/ReadVariableOp_1�!CBlock_1_BatchNorm/ReadVariableOp�#CBlock_1_BatchNorm/ReadVariableOp_1�)CBlock_1_SepConv2D/BiasAdd/ReadVariableOp�2CBlock_1_SepConv2D/separable_conv2d/ReadVariableOp�4CBlock_1_SepConv2D/separable_conv2d/ReadVariableOp_1�2CBlock_2_BatchNorm/FusedBatchNormV3/ReadVariableOp�4CBlock_2_BatchNorm/FusedBatchNormV3/ReadVariableOp_1�!CBlock_2_BatchNorm/ReadVariableOp�#CBlock_2_BatchNorm/ReadVariableOp_1�)CBlock_2_SepConv2D/BiasAdd/ReadVariableOp�2CBlock_2_SepConv2D/separable_conv2d/ReadVariableOp�4CBlock_2_SepConv2D/separable_conv2d/ReadVariableOp_1�2CBlock_3_BatchNorm/FusedBatchNormV3/ReadVariableOp�4CBlock_3_BatchNorm/FusedBatchNormV3/ReadVariableOp_1�!CBlock_3_BatchNorm/ReadVariableOp�#CBlock_3_BatchNorm/ReadVariableOp_1�)CBlock_3_SepConv2D/BiasAdd/ReadVariableOp�2CBlock_3_SepConv2D/separable_conv2d/ReadVariableOp�4CBlock_3_SepConv2D/separable_conv2d/ReadVariableOp_1�2CBlock_4_BatchNorm/FusedBatchNormV3/ReadVariableOp�4CBlock_4_BatchNorm/FusedBatchNormV3/ReadVariableOp_1�!CBlock_4_BatchNorm/ReadVariableOp�#CBlock_4_BatchNorm/ReadVariableOp_1�)CBlock_4_SepConv2D/BiasAdd/ReadVariableOp�2CBlock_4_SepConv2D/separable_conv2d/ReadVariableOp�4CBlock_4_SepConv2D/separable_conv2d/ReadVariableOp_1�Conv0/BiasAdd/ReadVariableOp�Conv0/Conv2D/ReadVariableOp�dense_8/BiasAdd/ReadVariableOp�dense_8/MatMul/ReadVariableOp�
Conv0/Conv2D/ReadVariableOpReadVariableOp$conv0_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
Conv0/Conv2DConv2Dinputs#Conv0/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������$,*
paddingSAME*
strides
~
Conv0/BiasAdd/ReadVariableOpReadVariableOp%conv0_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
Conv0/BiasAddBiasAddConv0/Conv2D:output:0$Conv0/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������$,�
2CBlock_1_SepConv2D/separable_conv2d/ReadVariableOpReadVariableOp;cblock_1_sepconv2d_separable_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
4CBlock_1_SepConv2D/separable_conv2d/ReadVariableOp_1ReadVariableOp=cblock_1_sepconv2d_separable_conv2d_readvariableop_1_resource*&
_output_shapes
:*
dtype0�
)CBlock_1_SepConv2D/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            �
1CBlock_1_SepConv2D/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      �
-CBlock_1_SepConv2D/separable_conv2d/depthwiseDepthwiseConv2dNativeConv0/BiasAdd:output:0:CBlock_1_SepConv2D/separable_conv2d/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
�
#CBlock_1_SepConv2D/separable_conv2dConv2D6CBlock_1_SepConv2D/separable_conv2d/depthwise:output:0<CBlock_1_SepConv2D/separable_conv2d/ReadVariableOp_1:value:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
�
)CBlock_1_SepConv2D/BiasAdd/ReadVariableOpReadVariableOp2cblock_1_sepconv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
CBlock_1_SepConv2D/BiasAddBiasAdd,CBlock_1_SepConv2D/separable_conv2d:output:01CBlock_1_SepConv2D/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:����������
!CBlock_1_BatchNorm/ReadVariableOpReadVariableOp*cblock_1_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0�
#CBlock_1_BatchNorm/ReadVariableOp_1ReadVariableOp,cblock_1_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0�
2CBlock_1_BatchNorm/FusedBatchNormV3/ReadVariableOpReadVariableOp;cblock_1_batchnorm_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0�
4CBlock_1_BatchNorm/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp=cblock_1_batchnorm_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0�
#CBlock_1_BatchNorm/FusedBatchNormV3FusedBatchNormV3#CBlock_1_SepConv2D/BiasAdd:output:0)CBlock_1_BatchNorm/ReadVariableOp:value:0+CBlock_1_BatchNorm/ReadVariableOp_1:value:0:CBlock_1_BatchNorm/FusedBatchNormV3/ReadVariableOp:value:0<CBlock_1_BatchNorm/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������:::::*
epsilon%o�:*
is_training( 
CBlock_1_ReLu/Relu6Relu6'CBlock_1_BatchNorm/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:����������
2CBlock_2_SepConv2D/separable_conv2d/ReadVariableOpReadVariableOp;cblock_2_sepconv2d_separable_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
4CBlock_2_SepConv2D/separable_conv2d/ReadVariableOp_1ReadVariableOp=cblock_2_sepconv2d_separable_conv2d_readvariableop_1_resource*&
_output_shapes
:1*
dtype0�
)CBlock_2_SepConv2D/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            �
1CBlock_2_SepConv2D/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      �
-CBlock_2_SepConv2D/separable_conv2d/depthwiseDepthwiseConv2dNative!CBlock_1_ReLu/Relu6:activations:0:CBlock_2_SepConv2D/separable_conv2d/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������	*
paddingSAME*
strides
�
#CBlock_2_SepConv2D/separable_conv2dConv2D6CBlock_2_SepConv2D/separable_conv2d/depthwise:output:0<CBlock_2_SepConv2D/separable_conv2d/ReadVariableOp_1:value:0*
T0*/
_output_shapes
:���������	1*
paddingVALID*
strides
�
)CBlock_2_SepConv2D/BiasAdd/ReadVariableOpReadVariableOp2cblock_2_sepconv2d_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype0�
CBlock_2_SepConv2D/BiasAddBiasAdd,CBlock_2_SepConv2D/separable_conv2d:output:01CBlock_2_SepConv2D/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������	1�
!CBlock_2_BatchNorm/ReadVariableOpReadVariableOp*cblock_2_batchnorm_readvariableop_resource*
_output_shapes
:1*
dtype0�
#CBlock_2_BatchNorm/ReadVariableOp_1ReadVariableOp,cblock_2_batchnorm_readvariableop_1_resource*
_output_shapes
:1*
dtype0�
2CBlock_2_BatchNorm/FusedBatchNormV3/ReadVariableOpReadVariableOp;cblock_2_batchnorm_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:1*
dtype0�
4CBlock_2_BatchNorm/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp=cblock_2_batchnorm_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:1*
dtype0�
#CBlock_2_BatchNorm/FusedBatchNormV3FusedBatchNormV3#CBlock_2_SepConv2D/BiasAdd:output:0)CBlock_2_BatchNorm/ReadVariableOp:value:0+CBlock_2_BatchNorm/ReadVariableOp_1:value:0:CBlock_2_BatchNorm/FusedBatchNormV3/ReadVariableOp:value:0<CBlock_2_BatchNorm/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������	1:1:1:1:1:*
epsilon%o�:*
is_training( 
CBlock_2_ReLu/Relu6Relu6'CBlock_2_BatchNorm/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:���������	1�
2CBlock_3_SepConv2D/separable_conv2d/ReadVariableOpReadVariableOp;cblock_3_sepconv2d_separable_conv2d_readvariableop_resource*&
_output_shapes
:1*
dtype0�
4CBlock_3_SepConv2D/separable_conv2d/ReadVariableOp_1ReadVariableOp=cblock_3_sepconv2d_separable_conv2d_readvariableop_1_resource*&
_output_shapes
:1J*
dtype0�
)CBlock_3_SepConv2D/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      1      �
1CBlock_3_SepConv2D/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      �
-CBlock_3_SepConv2D/separable_conv2d/depthwiseDepthwiseConv2dNative!CBlock_2_ReLu/Relu6:activations:0:CBlock_3_SepConv2D/separable_conv2d/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������1*
paddingSAME*
strides
�
#CBlock_3_SepConv2D/separable_conv2dConv2D6CBlock_3_SepConv2D/separable_conv2d/depthwise:output:0<CBlock_3_SepConv2D/separable_conv2d/ReadVariableOp_1:value:0*
T0*/
_output_shapes
:���������J*
paddingVALID*
strides
�
)CBlock_3_SepConv2D/BiasAdd/ReadVariableOpReadVariableOp2cblock_3_sepconv2d_biasadd_readvariableop_resource*
_output_shapes
:J*
dtype0�
CBlock_3_SepConv2D/BiasAddBiasAdd,CBlock_3_SepConv2D/separable_conv2d:output:01CBlock_3_SepConv2D/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������J�
!CBlock_3_BatchNorm/ReadVariableOpReadVariableOp*cblock_3_batchnorm_readvariableop_resource*
_output_shapes
:J*
dtype0�
#CBlock_3_BatchNorm/ReadVariableOp_1ReadVariableOp,cblock_3_batchnorm_readvariableop_1_resource*
_output_shapes
:J*
dtype0�
2CBlock_3_BatchNorm/FusedBatchNormV3/ReadVariableOpReadVariableOp;cblock_3_batchnorm_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:J*
dtype0�
4CBlock_3_BatchNorm/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp=cblock_3_batchnorm_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:J*
dtype0�
#CBlock_3_BatchNorm/FusedBatchNormV3FusedBatchNormV3#CBlock_3_SepConv2D/BiasAdd:output:0)CBlock_3_BatchNorm/ReadVariableOp:value:0+CBlock_3_BatchNorm/ReadVariableOp_1:value:0:CBlock_3_BatchNorm/FusedBatchNormV3/ReadVariableOp:value:0<CBlock_3_BatchNorm/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������J:J:J:J:J:*
epsilon%o�:*
is_training( 
CBlock_3_ReLu/Relu6Relu6'CBlock_3_BatchNorm/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:���������J�
2CBlock_4_SepConv2D/separable_conv2d/ReadVariableOpReadVariableOp;cblock_4_sepconv2d_separable_conv2d_readvariableop_resource*&
_output_shapes
:J*
dtype0�
4CBlock_4_SepConv2D/separable_conv2d/ReadVariableOp_1ReadVariableOp=cblock_4_sepconv2d_separable_conv2d_readvariableop_1_resource*&
_output_shapes
:Jc*
dtype0�
)CBlock_4_SepConv2D/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      J      �
1CBlock_4_SepConv2D/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      �
-CBlock_4_SepConv2D/separable_conv2d/depthwiseDepthwiseConv2dNative!CBlock_3_ReLu/Relu6:activations:0:CBlock_4_SepConv2D/separable_conv2d/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������J*
paddingSAME*
strides
�
#CBlock_4_SepConv2D/separable_conv2dConv2D6CBlock_4_SepConv2D/separable_conv2d/depthwise:output:0<CBlock_4_SepConv2D/separable_conv2d/ReadVariableOp_1:value:0*
T0*/
_output_shapes
:���������c*
paddingVALID*
strides
�
)CBlock_4_SepConv2D/BiasAdd/ReadVariableOpReadVariableOp2cblock_4_sepconv2d_biasadd_readvariableop_resource*
_output_shapes
:c*
dtype0�
CBlock_4_SepConv2D/BiasAddBiasAdd,CBlock_4_SepConv2D/separable_conv2d:output:01CBlock_4_SepConv2D/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������c�
!CBlock_4_BatchNorm/ReadVariableOpReadVariableOp*cblock_4_batchnorm_readvariableop_resource*
_output_shapes
:c*
dtype0�
#CBlock_4_BatchNorm/ReadVariableOp_1ReadVariableOp,cblock_4_batchnorm_readvariableop_1_resource*
_output_shapes
:c*
dtype0�
2CBlock_4_BatchNorm/FusedBatchNormV3/ReadVariableOpReadVariableOp;cblock_4_batchnorm_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:c*
dtype0�
4CBlock_4_BatchNorm/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp=cblock_4_batchnorm_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:c*
dtype0�
#CBlock_4_BatchNorm/FusedBatchNormV3FusedBatchNormV3#CBlock_4_SepConv2D/BiasAdd:output:0)CBlock_4_BatchNorm/ReadVariableOp:value:0+CBlock_4_BatchNorm/ReadVariableOp_1:value:0:CBlock_4_BatchNorm/FusedBatchNormV3/ReadVariableOp:value:0<CBlock_4_BatchNorm/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������c:c:c:c:c:*
epsilon%o�:*
is_training( 
CBlock_4_ReLu/Relu6Relu6'CBlock_4_BatchNorm/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:���������c�
1global_average_pooling2d_8/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      �
global_average_pooling2d_8/MeanMean!CBlock_4_ReLu/Relu6:activations:0:global_average_pooling2d_8/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:���������c�
dense_8/MatMul/ReadVariableOpReadVariableOp&dense_8_matmul_readvariableop_resource*
_output_shapes

:c*
dtype0�
dense_8/MatMulMatMul(global_average_pooling2d_8/Mean:output:0%dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense_8/BiasAdd/ReadVariableOpReadVariableOp'dense_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_8/BiasAddBiasAdddense_8/MatMul:product:0&dense_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������f
dense_8/SoftmaxSoftmaxdense_8/BiasAdd:output:0*
T0*'
_output_shapes
:���������h
IdentityIdentitydense_8/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp3^CBlock_1_BatchNorm/FusedBatchNormV3/ReadVariableOp5^CBlock_1_BatchNorm/FusedBatchNormV3/ReadVariableOp_1"^CBlock_1_BatchNorm/ReadVariableOp$^CBlock_1_BatchNorm/ReadVariableOp_1*^CBlock_1_SepConv2D/BiasAdd/ReadVariableOp3^CBlock_1_SepConv2D/separable_conv2d/ReadVariableOp5^CBlock_1_SepConv2D/separable_conv2d/ReadVariableOp_13^CBlock_2_BatchNorm/FusedBatchNormV3/ReadVariableOp5^CBlock_2_BatchNorm/FusedBatchNormV3/ReadVariableOp_1"^CBlock_2_BatchNorm/ReadVariableOp$^CBlock_2_BatchNorm/ReadVariableOp_1*^CBlock_2_SepConv2D/BiasAdd/ReadVariableOp3^CBlock_2_SepConv2D/separable_conv2d/ReadVariableOp5^CBlock_2_SepConv2D/separable_conv2d/ReadVariableOp_13^CBlock_3_BatchNorm/FusedBatchNormV3/ReadVariableOp5^CBlock_3_BatchNorm/FusedBatchNormV3/ReadVariableOp_1"^CBlock_3_BatchNorm/ReadVariableOp$^CBlock_3_BatchNorm/ReadVariableOp_1*^CBlock_3_SepConv2D/BiasAdd/ReadVariableOp3^CBlock_3_SepConv2D/separable_conv2d/ReadVariableOp5^CBlock_3_SepConv2D/separable_conv2d/ReadVariableOp_13^CBlock_4_BatchNorm/FusedBatchNormV3/ReadVariableOp5^CBlock_4_BatchNorm/FusedBatchNormV3/ReadVariableOp_1"^CBlock_4_BatchNorm/ReadVariableOp$^CBlock_4_BatchNorm/ReadVariableOp_1*^CBlock_4_SepConv2D/BiasAdd/ReadVariableOp3^CBlock_4_SepConv2D/separable_conv2d/ReadVariableOp5^CBlock_4_SepConv2D/separable_conv2d/ReadVariableOp_1^Conv0/BiasAdd/ReadVariableOp^Conv0/Conv2D/ReadVariableOp^dense_8/BiasAdd/ReadVariableOp^dense_8/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*n
_input_shapes]
[:���������HX: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2l
4CBlock_1_BatchNorm/FusedBatchNormV3/ReadVariableOp_14CBlock_1_BatchNorm/FusedBatchNormV3/ReadVariableOp_12h
2CBlock_1_BatchNorm/FusedBatchNormV3/ReadVariableOp2CBlock_1_BatchNorm/FusedBatchNormV3/ReadVariableOp2J
#CBlock_1_BatchNorm/ReadVariableOp_1#CBlock_1_BatchNorm/ReadVariableOp_12F
!CBlock_1_BatchNorm/ReadVariableOp!CBlock_1_BatchNorm/ReadVariableOp2V
)CBlock_1_SepConv2D/BiasAdd/ReadVariableOp)CBlock_1_SepConv2D/BiasAdd/ReadVariableOp2l
4CBlock_1_SepConv2D/separable_conv2d/ReadVariableOp_14CBlock_1_SepConv2D/separable_conv2d/ReadVariableOp_12h
2CBlock_1_SepConv2D/separable_conv2d/ReadVariableOp2CBlock_1_SepConv2D/separable_conv2d/ReadVariableOp2l
4CBlock_2_BatchNorm/FusedBatchNormV3/ReadVariableOp_14CBlock_2_BatchNorm/FusedBatchNormV3/ReadVariableOp_12h
2CBlock_2_BatchNorm/FusedBatchNormV3/ReadVariableOp2CBlock_2_BatchNorm/FusedBatchNormV3/ReadVariableOp2J
#CBlock_2_BatchNorm/ReadVariableOp_1#CBlock_2_BatchNorm/ReadVariableOp_12F
!CBlock_2_BatchNorm/ReadVariableOp!CBlock_2_BatchNorm/ReadVariableOp2V
)CBlock_2_SepConv2D/BiasAdd/ReadVariableOp)CBlock_2_SepConv2D/BiasAdd/ReadVariableOp2l
4CBlock_2_SepConv2D/separable_conv2d/ReadVariableOp_14CBlock_2_SepConv2D/separable_conv2d/ReadVariableOp_12h
2CBlock_2_SepConv2D/separable_conv2d/ReadVariableOp2CBlock_2_SepConv2D/separable_conv2d/ReadVariableOp2l
4CBlock_3_BatchNorm/FusedBatchNormV3/ReadVariableOp_14CBlock_3_BatchNorm/FusedBatchNormV3/ReadVariableOp_12h
2CBlock_3_BatchNorm/FusedBatchNormV3/ReadVariableOp2CBlock_3_BatchNorm/FusedBatchNormV3/ReadVariableOp2J
#CBlock_3_BatchNorm/ReadVariableOp_1#CBlock_3_BatchNorm/ReadVariableOp_12F
!CBlock_3_BatchNorm/ReadVariableOp!CBlock_3_BatchNorm/ReadVariableOp2V
)CBlock_3_SepConv2D/BiasAdd/ReadVariableOp)CBlock_3_SepConv2D/BiasAdd/ReadVariableOp2l
4CBlock_3_SepConv2D/separable_conv2d/ReadVariableOp_14CBlock_3_SepConv2D/separable_conv2d/ReadVariableOp_12h
2CBlock_3_SepConv2D/separable_conv2d/ReadVariableOp2CBlock_3_SepConv2D/separable_conv2d/ReadVariableOp2l
4CBlock_4_BatchNorm/FusedBatchNormV3/ReadVariableOp_14CBlock_4_BatchNorm/FusedBatchNormV3/ReadVariableOp_12h
2CBlock_4_BatchNorm/FusedBatchNormV3/ReadVariableOp2CBlock_4_BatchNorm/FusedBatchNormV3/ReadVariableOp2J
#CBlock_4_BatchNorm/ReadVariableOp_1#CBlock_4_BatchNorm/ReadVariableOp_12F
!CBlock_4_BatchNorm/ReadVariableOp!CBlock_4_BatchNorm/ReadVariableOp2V
)CBlock_4_SepConv2D/BiasAdd/ReadVariableOp)CBlock_4_SepConv2D/BiasAdd/ReadVariableOp2l
4CBlock_4_SepConv2D/separable_conv2d/ReadVariableOp_14CBlock_4_SepConv2D/separable_conv2d/ReadVariableOp_12h
2CBlock_4_SepConv2D/separable_conv2d/ReadVariableOp2CBlock_4_SepConv2D/separable_conv2d/ReadVariableOp2<
Conv0/BiasAdd/ReadVariableOpConv0/BiasAdd/ReadVariableOp2:
Conv0/Conv2D/ReadVariableOpConv0/Conv2D/ReadVariableOp2@
dense_8/BiasAdd/ReadVariableOpdense_8/BiasAdd/ReadVariableOp2>
dense_8/MatMul/ReadVariableOpdense_8/MatMul/ReadVariableOp:W S
/
_output_shapes
:���������HX
 
_user_specified_nameinputs
�	
�
4__inference_CBlock_4_BatchNorm_layer_call_fn_1690635

inputs
unknown:c
	unknown_0:c
	unknown_1:c
	unknown_2:c
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������c*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *X
fSRQ
O__inference_CBlock_4_BatchNorm_layer_call_and_return_conditional_losses_1689068�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������c`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������c: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������c
 
_user_specified_nameinputs
�
�
O__inference_CBlock_1_BatchNorm_layer_call_and_return_conditional_losses_1690359

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������:::::*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+����������������������������
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������: : : : 2$
AssignNewValue_1AssignNewValue_12 
AssignNewValueAssignNewValue2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
K
/__inference_CBlock_2_ReLu_layer_call_fn_1690480

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������	1* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_CBlock_2_ReLu_layer_call_and_return_conditional_losses_1689172h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������	1"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������	1:W S
/
_output_shapes
:���������	1
 
_user_specified_nameinputs
�
K
/__inference_CBlock_4_ReLu_layer_call_fn_1690676

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������c* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_CBlock_4_ReLu_layer_call_and_return_conditional_losses_1689218h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������c"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������c:W S
/
_output_shapes
:���������c
 
_user_specified_nameinputs
�	
�
4__inference_CBlock_1_BatchNorm_layer_call_fn_1690328

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *X
fSRQ
O__inference_CBlock_1_BatchNorm_layer_call_and_return_conditional_losses_1688774�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
�
O__inference_CBlock_3_BatchNorm_layer_call_and_return_conditional_losses_1688958

inputs%
readvariableop_resource:J'
readvariableop_1_resource:J6
(fusedbatchnormv3_readvariableop_resource:J8
*fusedbatchnormv3_readvariableop_1_resource:J
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:J*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:J*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:J*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:J*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������J:J:J:J:J:*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������J�
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������J: : : : 2$
AssignNewValue_1AssignNewValue_12 
AssignNewValueAssignNewValue2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:i e
A
_output_shapes/
-:+���������������������������J
 
_user_specified_nameinputs
�
X
<__inference_global_average_pooling2d_8_layer_call_fn_1690686

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:������������������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *`
f[RY
W__inference_global_average_pooling2d_8_layer_call_and_return_conditional_losses_1689102i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
�
O__inference_CBlock_3_BatchNorm_layer_call_and_return_conditional_losses_1688976

inputs%
readvariableop_resource:J'
readvariableop_1_resource:J6
(fusedbatchnormv3_readvariableop_resource:J8
*fusedbatchnormv3_readvariableop_1_resource:J
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:J*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:J*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:J*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:J*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������J:J:J:J:J:*
epsilon%o�:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������J�
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������J: : : : 2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:i e
A
_output_shapes/
-:+���������������������������J
 
_user_specified_nameinputs
�
�
O__inference_CBlock_4_BatchNorm_layer_call_and_return_conditional_losses_1689068

inputs%
readvariableop_resource:c'
readvariableop_1_resource:c6
(fusedbatchnormv3_readvariableop_resource:c8
*fusedbatchnormv3_readvariableop_1_resource:c
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:c*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:c*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:c*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:c*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������c:c:c:c:c:*
epsilon%o�:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������c�
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������c: : : : 2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:i e
A
_output_shapes/
-:+���������������������������c
 
_user_specified_nameinputs
�
f
J__inference_CBlock_3_ReLu_layer_call_and_return_conditional_losses_1690583

inputs
identityP
Relu6Relu6inputs*
T0*/
_output_shapes
:���������Jc
IdentityIdentityRelu6:activations:0*
T0*/
_output_shapes
:���������J"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������J:W S
/
_output_shapes
:���������J
 
_user_specified_nameinputs
�
�
'__inference_Conv0_layer_call_fn_1690279

inputs!
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������$,*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_Conv0_layer_call_and_return_conditional_losses_1689122w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������$,`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������HX: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������HX
 
_user_specified_nameinputs
�
�
O__inference_CBlock_3_BatchNorm_layer_call_and_return_conditional_losses_1690573

inputs%
readvariableop_resource:J'
readvariableop_1_resource:J6
(fusedbatchnormv3_readvariableop_resource:J8
*fusedbatchnormv3_readvariableop_1_resource:J
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:J*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:J*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:J*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:J*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������J:J:J:J:J:*
epsilon%o�:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������J�
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������J: : : : 2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:i e
A
_output_shapes/
-:+���������������������������J
 
_user_specified_nameinputs
�
�
O__inference_CBlock_3_BatchNorm_layer_call_and_return_conditional_losses_1690555

inputs%
readvariableop_resource:J'
readvariableop_1_resource:J6
(fusedbatchnormv3_readvariableop_resource:J8
*fusedbatchnormv3_readvariableop_1_resource:J
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:J*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:J*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:J*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:J*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������J:J:J:J:J:*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������J�
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������J: : : : 2$
AssignNewValue_1AssignNewValue_12 
AssignNewValueAssignNewValue2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:i e
A
_output_shapes/
-:+���������������������������J
 
_user_specified_nameinputs
�

�
B__inference_Conv0_layer_call_and_return_conditional_losses_1689122

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������$,*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������$,g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:���������$,w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������HX: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������HX
 
_user_specified_nameinputs
��
� 
a__inference_jojo_n8_r72x88_a0.7_b4_g2.2_strides2_layer_call_and_return_conditional_losses_1690147

inputs>
$conv0_conv2d_readvariableop_resource:3
%conv0_biasadd_readvariableop_resource:U
;cblock_1_sepconv2d_separable_conv2d_readvariableop_resource:W
=cblock_1_sepconv2d_separable_conv2d_readvariableop_1_resource:@
2cblock_1_sepconv2d_biasadd_readvariableop_resource:8
*cblock_1_batchnorm_readvariableop_resource::
,cblock_1_batchnorm_readvariableop_1_resource:I
;cblock_1_batchnorm_fusedbatchnormv3_readvariableop_resource:K
=cblock_1_batchnorm_fusedbatchnormv3_readvariableop_1_resource:U
;cblock_2_sepconv2d_separable_conv2d_readvariableop_resource:W
=cblock_2_sepconv2d_separable_conv2d_readvariableop_1_resource:1@
2cblock_2_sepconv2d_biasadd_readvariableop_resource:18
*cblock_2_batchnorm_readvariableop_resource:1:
,cblock_2_batchnorm_readvariableop_1_resource:1I
;cblock_2_batchnorm_fusedbatchnormv3_readvariableop_resource:1K
=cblock_2_batchnorm_fusedbatchnormv3_readvariableop_1_resource:1U
;cblock_3_sepconv2d_separable_conv2d_readvariableop_resource:1W
=cblock_3_sepconv2d_separable_conv2d_readvariableop_1_resource:1J@
2cblock_3_sepconv2d_biasadd_readvariableop_resource:J8
*cblock_3_batchnorm_readvariableop_resource:J:
,cblock_3_batchnorm_readvariableop_1_resource:JI
;cblock_3_batchnorm_fusedbatchnormv3_readvariableop_resource:JK
=cblock_3_batchnorm_fusedbatchnormv3_readvariableop_1_resource:JU
;cblock_4_sepconv2d_separable_conv2d_readvariableop_resource:JW
=cblock_4_sepconv2d_separable_conv2d_readvariableop_1_resource:Jc@
2cblock_4_sepconv2d_biasadd_readvariableop_resource:c8
*cblock_4_batchnorm_readvariableop_resource:c:
,cblock_4_batchnorm_readvariableop_1_resource:cI
;cblock_4_batchnorm_fusedbatchnormv3_readvariableop_resource:cK
=cblock_4_batchnorm_fusedbatchnormv3_readvariableop_1_resource:c8
&dense_8_matmul_readvariableop_resource:c5
'dense_8_biasadd_readvariableop_resource:
identity��!CBlock_1_BatchNorm/AssignNewValue�#CBlock_1_BatchNorm/AssignNewValue_1�2CBlock_1_BatchNorm/FusedBatchNormV3/ReadVariableOp�4CBlock_1_BatchNorm/FusedBatchNormV3/ReadVariableOp_1�!CBlock_1_BatchNorm/ReadVariableOp�#CBlock_1_BatchNorm/ReadVariableOp_1�)CBlock_1_SepConv2D/BiasAdd/ReadVariableOp�2CBlock_1_SepConv2D/separable_conv2d/ReadVariableOp�4CBlock_1_SepConv2D/separable_conv2d/ReadVariableOp_1�!CBlock_2_BatchNorm/AssignNewValue�#CBlock_2_BatchNorm/AssignNewValue_1�2CBlock_2_BatchNorm/FusedBatchNormV3/ReadVariableOp�4CBlock_2_BatchNorm/FusedBatchNormV3/ReadVariableOp_1�!CBlock_2_BatchNorm/ReadVariableOp�#CBlock_2_BatchNorm/ReadVariableOp_1�)CBlock_2_SepConv2D/BiasAdd/ReadVariableOp�2CBlock_2_SepConv2D/separable_conv2d/ReadVariableOp�4CBlock_2_SepConv2D/separable_conv2d/ReadVariableOp_1�!CBlock_3_BatchNorm/AssignNewValue�#CBlock_3_BatchNorm/AssignNewValue_1�2CBlock_3_BatchNorm/FusedBatchNormV3/ReadVariableOp�4CBlock_3_BatchNorm/FusedBatchNormV3/ReadVariableOp_1�!CBlock_3_BatchNorm/ReadVariableOp�#CBlock_3_BatchNorm/ReadVariableOp_1�)CBlock_3_SepConv2D/BiasAdd/ReadVariableOp�2CBlock_3_SepConv2D/separable_conv2d/ReadVariableOp�4CBlock_3_SepConv2D/separable_conv2d/ReadVariableOp_1�!CBlock_4_BatchNorm/AssignNewValue�#CBlock_4_BatchNorm/AssignNewValue_1�2CBlock_4_BatchNorm/FusedBatchNormV3/ReadVariableOp�4CBlock_4_BatchNorm/FusedBatchNormV3/ReadVariableOp_1�!CBlock_4_BatchNorm/ReadVariableOp�#CBlock_4_BatchNorm/ReadVariableOp_1�)CBlock_4_SepConv2D/BiasAdd/ReadVariableOp�2CBlock_4_SepConv2D/separable_conv2d/ReadVariableOp�4CBlock_4_SepConv2D/separable_conv2d/ReadVariableOp_1�Conv0/BiasAdd/ReadVariableOp�Conv0/Conv2D/ReadVariableOp�dense_8/BiasAdd/ReadVariableOp�dense_8/MatMul/ReadVariableOp�
Conv0/Conv2D/ReadVariableOpReadVariableOp$conv0_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
Conv0/Conv2DConv2Dinputs#Conv0/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������$,*
paddingSAME*
strides
~
Conv0/BiasAdd/ReadVariableOpReadVariableOp%conv0_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
Conv0/BiasAddBiasAddConv0/Conv2D:output:0$Conv0/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������$,�
2CBlock_1_SepConv2D/separable_conv2d/ReadVariableOpReadVariableOp;cblock_1_sepconv2d_separable_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
4CBlock_1_SepConv2D/separable_conv2d/ReadVariableOp_1ReadVariableOp=cblock_1_sepconv2d_separable_conv2d_readvariableop_1_resource*&
_output_shapes
:*
dtype0�
)CBlock_1_SepConv2D/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            �
1CBlock_1_SepConv2D/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      �
-CBlock_1_SepConv2D/separable_conv2d/depthwiseDepthwiseConv2dNativeConv0/BiasAdd:output:0:CBlock_1_SepConv2D/separable_conv2d/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
�
#CBlock_1_SepConv2D/separable_conv2dConv2D6CBlock_1_SepConv2D/separable_conv2d/depthwise:output:0<CBlock_1_SepConv2D/separable_conv2d/ReadVariableOp_1:value:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
�
)CBlock_1_SepConv2D/BiasAdd/ReadVariableOpReadVariableOp2cblock_1_sepconv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
CBlock_1_SepConv2D/BiasAddBiasAdd,CBlock_1_SepConv2D/separable_conv2d:output:01CBlock_1_SepConv2D/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:����������
!CBlock_1_BatchNorm/ReadVariableOpReadVariableOp*cblock_1_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0�
#CBlock_1_BatchNorm/ReadVariableOp_1ReadVariableOp,cblock_1_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0�
2CBlock_1_BatchNorm/FusedBatchNormV3/ReadVariableOpReadVariableOp;cblock_1_batchnorm_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0�
4CBlock_1_BatchNorm/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp=cblock_1_batchnorm_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0�
#CBlock_1_BatchNorm/FusedBatchNormV3FusedBatchNormV3#CBlock_1_SepConv2D/BiasAdd:output:0)CBlock_1_BatchNorm/ReadVariableOp:value:0+CBlock_1_BatchNorm/ReadVariableOp_1:value:0:CBlock_1_BatchNorm/FusedBatchNormV3/ReadVariableOp:value:0<CBlock_1_BatchNorm/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������:::::*
epsilon%o�:*
exponential_avg_factor%
�#<�
!CBlock_1_BatchNorm/AssignNewValueAssignVariableOp;cblock_1_batchnorm_fusedbatchnormv3_readvariableop_resource0CBlock_1_BatchNorm/FusedBatchNormV3:batch_mean:03^CBlock_1_BatchNorm/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
#CBlock_1_BatchNorm/AssignNewValue_1AssignVariableOp=cblock_1_batchnorm_fusedbatchnormv3_readvariableop_1_resource4CBlock_1_BatchNorm/FusedBatchNormV3:batch_variance:05^CBlock_1_BatchNorm/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(
CBlock_1_ReLu/Relu6Relu6'CBlock_1_BatchNorm/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:����������
2CBlock_2_SepConv2D/separable_conv2d/ReadVariableOpReadVariableOp;cblock_2_sepconv2d_separable_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
4CBlock_2_SepConv2D/separable_conv2d/ReadVariableOp_1ReadVariableOp=cblock_2_sepconv2d_separable_conv2d_readvariableop_1_resource*&
_output_shapes
:1*
dtype0�
)CBlock_2_SepConv2D/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            �
1CBlock_2_SepConv2D/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      �
-CBlock_2_SepConv2D/separable_conv2d/depthwiseDepthwiseConv2dNative!CBlock_1_ReLu/Relu6:activations:0:CBlock_2_SepConv2D/separable_conv2d/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������	*
paddingSAME*
strides
�
#CBlock_2_SepConv2D/separable_conv2dConv2D6CBlock_2_SepConv2D/separable_conv2d/depthwise:output:0<CBlock_2_SepConv2D/separable_conv2d/ReadVariableOp_1:value:0*
T0*/
_output_shapes
:���������	1*
paddingVALID*
strides
�
)CBlock_2_SepConv2D/BiasAdd/ReadVariableOpReadVariableOp2cblock_2_sepconv2d_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype0�
CBlock_2_SepConv2D/BiasAddBiasAdd,CBlock_2_SepConv2D/separable_conv2d:output:01CBlock_2_SepConv2D/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������	1�
!CBlock_2_BatchNorm/ReadVariableOpReadVariableOp*cblock_2_batchnorm_readvariableop_resource*
_output_shapes
:1*
dtype0�
#CBlock_2_BatchNorm/ReadVariableOp_1ReadVariableOp,cblock_2_batchnorm_readvariableop_1_resource*
_output_shapes
:1*
dtype0�
2CBlock_2_BatchNorm/FusedBatchNormV3/ReadVariableOpReadVariableOp;cblock_2_batchnorm_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:1*
dtype0�
4CBlock_2_BatchNorm/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp=cblock_2_batchnorm_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:1*
dtype0�
#CBlock_2_BatchNorm/FusedBatchNormV3FusedBatchNormV3#CBlock_2_SepConv2D/BiasAdd:output:0)CBlock_2_BatchNorm/ReadVariableOp:value:0+CBlock_2_BatchNorm/ReadVariableOp_1:value:0:CBlock_2_BatchNorm/FusedBatchNormV3/ReadVariableOp:value:0<CBlock_2_BatchNorm/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������	1:1:1:1:1:*
epsilon%o�:*
exponential_avg_factor%
�#<�
!CBlock_2_BatchNorm/AssignNewValueAssignVariableOp;cblock_2_batchnorm_fusedbatchnormv3_readvariableop_resource0CBlock_2_BatchNorm/FusedBatchNormV3:batch_mean:03^CBlock_2_BatchNorm/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
#CBlock_2_BatchNorm/AssignNewValue_1AssignVariableOp=cblock_2_batchnorm_fusedbatchnormv3_readvariableop_1_resource4CBlock_2_BatchNorm/FusedBatchNormV3:batch_variance:05^CBlock_2_BatchNorm/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(
CBlock_2_ReLu/Relu6Relu6'CBlock_2_BatchNorm/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:���������	1�
2CBlock_3_SepConv2D/separable_conv2d/ReadVariableOpReadVariableOp;cblock_3_sepconv2d_separable_conv2d_readvariableop_resource*&
_output_shapes
:1*
dtype0�
4CBlock_3_SepConv2D/separable_conv2d/ReadVariableOp_1ReadVariableOp=cblock_3_sepconv2d_separable_conv2d_readvariableop_1_resource*&
_output_shapes
:1J*
dtype0�
)CBlock_3_SepConv2D/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      1      �
1CBlock_3_SepConv2D/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      �
-CBlock_3_SepConv2D/separable_conv2d/depthwiseDepthwiseConv2dNative!CBlock_2_ReLu/Relu6:activations:0:CBlock_3_SepConv2D/separable_conv2d/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������1*
paddingSAME*
strides
�
#CBlock_3_SepConv2D/separable_conv2dConv2D6CBlock_3_SepConv2D/separable_conv2d/depthwise:output:0<CBlock_3_SepConv2D/separable_conv2d/ReadVariableOp_1:value:0*
T0*/
_output_shapes
:���������J*
paddingVALID*
strides
�
)CBlock_3_SepConv2D/BiasAdd/ReadVariableOpReadVariableOp2cblock_3_sepconv2d_biasadd_readvariableop_resource*
_output_shapes
:J*
dtype0�
CBlock_3_SepConv2D/BiasAddBiasAdd,CBlock_3_SepConv2D/separable_conv2d:output:01CBlock_3_SepConv2D/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������J�
!CBlock_3_BatchNorm/ReadVariableOpReadVariableOp*cblock_3_batchnorm_readvariableop_resource*
_output_shapes
:J*
dtype0�
#CBlock_3_BatchNorm/ReadVariableOp_1ReadVariableOp,cblock_3_batchnorm_readvariableop_1_resource*
_output_shapes
:J*
dtype0�
2CBlock_3_BatchNorm/FusedBatchNormV3/ReadVariableOpReadVariableOp;cblock_3_batchnorm_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:J*
dtype0�
4CBlock_3_BatchNorm/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp=cblock_3_batchnorm_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:J*
dtype0�
#CBlock_3_BatchNorm/FusedBatchNormV3FusedBatchNormV3#CBlock_3_SepConv2D/BiasAdd:output:0)CBlock_3_BatchNorm/ReadVariableOp:value:0+CBlock_3_BatchNorm/ReadVariableOp_1:value:0:CBlock_3_BatchNorm/FusedBatchNormV3/ReadVariableOp:value:0<CBlock_3_BatchNorm/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������J:J:J:J:J:*
epsilon%o�:*
exponential_avg_factor%
�#<�
!CBlock_3_BatchNorm/AssignNewValueAssignVariableOp;cblock_3_batchnorm_fusedbatchnormv3_readvariableop_resource0CBlock_3_BatchNorm/FusedBatchNormV3:batch_mean:03^CBlock_3_BatchNorm/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
#CBlock_3_BatchNorm/AssignNewValue_1AssignVariableOp=cblock_3_batchnorm_fusedbatchnormv3_readvariableop_1_resource4CBlock_3_BatchNorm/FusedBatchNormV3:batch_variance:05^CBlock_3_BatchNorm/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(
CBlock_3_ReLu/Relu6Relu6'CBlock_3_BatchNorm/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:���������J�
2CBlock_4_SepConv2D/separable_conv2d/ReadVariableOpReadVariableOp;cblock_4_sepconv2d_separable_conv2d_readvariableop_resource*&
_output_shapes
:J*
dtype0�
4CBlock_4_SepConv2D/separable_conv2d/ReadVariableOp_1ReadVariableOp=cblock_4_sepconv2d_separable_conv2d_readvariableop_1_resource*&
_output_shapes
:Jc*
dtype0�
)CBlock_4_SepConv2D/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      J      �
1CBlock_4_SepConv2D/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      �
-CBlock_4_SepConv2D/separable_conv2d/depthwiseDepthwiseConv2dNative!CBlock_3_ReLu/Relu6:activations:0:CBlock_4_SepConv2D/separable_conv2d/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������J*
paddingSAME*
strides
�
#CBlock_4_SepConv2D/separable_conv2dConv2D6CBlock_4_SepConv2D/separable_conv2d/depthwise:output:0<CBlock_4_SepConv2D/separable_conv2d/ReadVariableOp_1:value:0*
T0*/
_output_shapes
:���������c*
paddingVALID*
strides
�
)CBlock_4_SepConv2D/BiasAdd/ReadVariableOpReadVariableOp2cblock_4_sepconv2d_biasadd_readvariableop_resource*
_output_shapes
:c*
dtype0�
CBlock_4_SepConv2D/BiasAddBiasAdd,CBlock_4_SepConv2D/separable_conv2d:output:01CBlock_4_SepConv2D/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������c�
!CBlock_4_BatchNorm/ReadVariableOpReadVariableOp*cblock_4_batchnorm_readvariableop_resource*
_output_shapes
:c*
dtype0�
#CBlock_4_BatchNorm/ReadVariableOp_1ReadVariableOp,cblock_4_batchnorm_readvariableop_1_resource*
_output_shapes
:c*
dtype0�
2CBlock_4_BatchNorm/FusedBatchNormV3/ReadVariableOpReadVariableOp;cblock_4_batchnorm_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:c*
dtype0�
4CBlock_4_BatchNorm/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp=cblock_4_batchnorm_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:c*
dtype0�
#CBlock_4_BatchNorm/FusedBatchNormV3FusedBatchNormV3#CBlock_4_SepConv2D/BiasAdd:output:0)CBlock_4_BatchNorm/ReadVariableOp:value:0+CBlock_4_BatchNorm/ReadVariableOp_1:value:0:CBlock_4_BatchNorm/FusedBatchNormV3/ReadVariableOp:value:0<CBlock_4_BatchNorm/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������c:c:c:c:c:*
epsilon%o�:*
exponential_avg_factor%
�#<�
!CBlock_4_BatchNorm/AssignNewValueAssignVariableOp;cblock_4_batchnorm_fusedbatchnormv3_readvariableop_resource0CBlock_4_BatchNorm/FusedBatchNormV3:batch_mean:03^CBlock_4_BatchNorm/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
#CBlock_4_BatchNorm/AssignNewValue_1AssignVariableOp=cblock_4_batchnorm_fusedbatchnormv3_readvariableop_1_resource4CBlock_4_BatchNorm/FusedBatchNormV3:batch_variance:05^CBlock_4_BatchNorm/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(
CBlock_4_ReLu/Relu6Relu6'CBlock_4_BatchNorm/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:���������c�
1global_average_pooling2d_8/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      �
global_average_pooling2d_8/MeanMean!CBlock_4_ReLu/Relu6:activations:0:global_average_pooling2d_8/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:���������c�
dense_8/MatMul/ReadVariableOpReadVariableOp&dense_8_matmul_readvariableop_resource*
_output_shapes

:c*
dtype0�
dense_8/MatMulMatMul(global_average_pooling2d_8/Mean:output:0%dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense_8/BiasAdd/ReadVariableOpReadVariableOp'dense_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_8/BiasAddBiasAdddense_8/MatMul:product:0&dense_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������f
dense_8/SoftmaxSoftmaxdense_8/BiasAdd:output:0*
T0*'
_output_shapes
:���������h
IdentityIdentitydense_8/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp"^CBlock_1_BatchNorm/AssignNewValue$^CBlock_1_BatchNorm/AssignNewValue_13^CBlock_1_BatchNorm/FusedBatchNormV3/ReadVariableOp5^CBlock_1_BatchNorm/FusedBatchNormV3/ReadVariableOp_1"^CBlock_1_BatchNorm/ReadVariableOp$^CBlock_1_BatchNorm/ReadVariableOp_1*^CBlock_1_SepConv2D/BiasAdd/ReadVariableOp3^CBlock_1_SepConv2D/separable_conv2d/ReadVariableOp5^CBlock_1_SepConv2D/separable_conv2d/ReadVariableOp_1"^CBlock_2_BatchNorm/AssignNewValue$^CBlock_2_BatchNorm/AssignNewValue_13^CBlock_2_BatchNorm/FusedBatchNormV3/ReadVariableOp5^CBlock_2_BatchNorm/FusedBatchNormV3/ReadVariableOp_1"^CBlock_2_BatchNorm/ReadVariableOp$^CBlock_2_BatchNorm/ReadVariableOp_1*^CBlock_2_SepConv2D/BiasAdd/ReadVariableOp3^CBlock_2_SepConv2D/separable_conv2d/ReadVariableOp5^CBlock_2_SepConv2D/separable_conv2d/ReadVariableOp_1"^CBlock_3_BatchNorm/AssignNewValue$^CBlock_3_BatchNorm/AssignNewValue_13^CBlock_3_BatchNorm/FusedBatchNormV3/ReadVariableOp5^CBlock_3_BatchNorm/FusedBatchNormV3/ReadVariableOp_1"^CBlock_3_BatchNorm/ReadVariableOp$^CBlock_3_BatchNorm/ReadVariableOp_1*^CBlock_3_SepConv2D/BiasAdd/ReadVariableOp3^CBlock_3_SepConv2D/separable_conv2d/ReadVariableOp5^CBlock_3_SepConv2D/separable_conv2d/ReadVariableOp_1"^CBlock_4_BatchNorm/AssignNewValue$^CBlock_4_BatchNorm/AssignNewValue_13^CBlock_4_BatchNorm/FusedBatchNormV3/ReadVariableOp5^CBlock_4_BatchNorm/FusedBatchNormV3/ReadVariableOp_1"^CBlock_4_BatchNorm/ReadVariableOp$^CBlock_4_BatchNorm/ReadVariableOp_1*^CBlock_4_SepConv2D/BiasAdd/ReadVariableOp3^CBlock_4_SepConv2D/separable_conv2d/ReadVariableOp5^CBlock_4_SepConv2D/separable_conv2d/ReadVariableOp_1^Conv0/BiasAdd/ReadVariableOp^Conv0/Conv2D/ReadVariableOp^dense_8/BiasAdd/ReadVariableOp^dense_8/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*n
_input_shapes]
[:���������HX: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2J
#CBlock_1_BatchNorm/AssignNewValue_1#CBlock_1_BatchNorm/AssignNewValue_12F
!CBlock_1_BatchNorm/AssignNewValue!CBlock_1_BatchNorm/AssignNewValue2l
4CBlock_1_BatchNorm/FusedBatchNormV3/ReadVariableOp_14CBlock_1_BatchNorm/FusedBatchNormV3/ReadVariableOp_12h
2CBlock_1_BatchNorm/FusedBatchNormV3/ReadVariableOp2CBlock_1_BatchNorm/FusedBatchNormV3/ReadVariableOp2J
#CBlock_1_BatchNorm/ReadVariableOp_1#CBlock_1_BatchNorm/ReadVariableOp_12F
!CBlock_1_BatchNorm/ReadVariableOp!CBlock_1_BatchNorm/ReadVariableOp2V
)CBlock_1_SepConv2D/BiasAdd/ReadVariableOp)CBlock_1_SepConv2D/BiasAdd/ReadVariableOp2l
4CBlock_1_SepConv2D/separable_conv2d/ReadVariableOp_14CBlock_1_SepConv2D/separable_conv2d/ReadVariableOp_12h
2CBlock_1_SepConv2D/separable_conv2d/ReadVariableOp2CBlock_1_SepConv2D/separable_conv2d/ReadVariableOp2J
#CBlock_2_BatchNorm/AssignNewValue_1#CBlock_2_BatchNorm/AssignNewValue_12F
!CBlock_2_BatchNorm/AssignNewValue!CBlock_2_BatchNorm/AssignNewValue2l
4CBlock_2_BatchNorm/FusedBatchNormV3/ReadVariableOp_14CBlock_2_BatchNorm/FusedBatchNormV3/ReadVariableOp_12h
2CBlock_2_BatchNorm/FusedBatchNormV3/ReadVariableOp2CBlock_2_BatchNorm/FusedBatchNormV3/ReadVariableOp2J
#CBlock_2_BatchNorm/ReadVariableOp_1#CBlock_2_BatchNorm/ReadVariableOp_12F
!CBlock_2_BatchNorm/ReadVariableOp!CBlock_2_BatchNorm/ReadVariableOp2V
)CBlock_2_SepConv2D/BiasAdd/ReadVariableOp)CBlock_2_SepConv2D/BiasAdd/ReadVariableOp2l
4CBlock_2_SepConv2D/separable_conv2d/ReadVariableOp_14CBlock_2_SepConv2D/separable_conv2d/ReadVariableOp_12h
2CBlock_2_SepConv2D/separable_conv2d/ReadVariableOp2CBlock_2_SepConv2D/separable_conv2d/ReadVariableOp2J
#CBlock_3_BatchNorm/AssignNewValue_1#CBlock_3_BatchNorm/AssignNewValue_12F
!CBlock_3_BatchNorm/AssignNewValue!CBlock_3_BatchNorm/AssignNewValue2l
4CBlock_3_BatchNorm/FusedBatchNormV3/ReadVariableOp_14CBlock_3_BatchNorm/FusedBatchNormV3/ReadVariableOp_12h
2CBlock_3_BatchNorm/FusedBatchNormV3/ReadVariableOp2CBlock_3_BatchNorm/FusedBatchNormV3/ReadVariableOp2J
#CBlock_3_BatchNorm/ReadVariableOp_1#CBlock_3_BatchNorm/ReadVariableOp_12F
!CBlock_3_BatchNorm/ReadVariableOp!CBlock_3_BatchNorm/ReadVariableOp2V
)CBlock_3_SepConv2D/BiasAdd/ReadVariableOp)CBlock_3_SepConv2D/BiasAdd/ReadVariableOp2l
4CBlock_3_SepConv2D/separable_conv2d/ReadVariableOp_14CBlock_3_SepConv2D/separable_conv2d/ReadVariableOp_12h
2CBlock_3_SepConv2D/separable_conv2d/ReadVariableOp2CBlock_3_SepConv2D/separable_conv2d/ReadVariableOp2J
#CBlock_4_BatchNorm/AssignNewValue_1#CBlock_4_BatchNorm/AssignNewValue_12F
!CBlock_4_BatchNorm/AssignNewValue!CBlock_4_BatchNorm/AssignNewValue2l
4CBlock_4_BatchNorm/FusedBatchNormV3/ReadVariableOp_14CBlock_4_BatchNorm/FusedBatchNormV3/ReadVariableOp_12h
2CBlock_4_BatchNorm/FusedBatchNormV3/ReadVariableOp2CBlock_4_BatchNorm/FusedBatchNormV3/ReadVariableOp2J
#CBlock_4_BatchNorm/ReadVariableOp_1#CBlock_4_BatchNorm/ReadVariableOp_12F
!CBlock_4_BatchNorm/ReadVariableOp!CBlock_4_BatchNorm/ReadVariableOp2V
)CBlock_4_SepConv2D/BiasAdd/ReadVariableOp)CBlock_4_SepConv2D/BiasAdd/ReadVariableOp2l
4CBlock_4_SepConv2D/separable_conv2d/ReadVariableOp_14CBlock_4_SepConv2D/separable_conv2d/ReadVariableOp_12h
2CBlock_4_SepConv2D/separable_conv2d/ReadVariableOp2CBlock_4_SepConv2D/separable_conv2d/ReadVariableOp2<
Conv0/BiasAdd/ReadVariableOpConv0/BiasAdd/ReadVariableOp2:
Conv0/Conv2D/ReadVariableOpConv0/Conv2D/ReadVariableOp2@
dense_8/BiasAdd/ReadVariableOpdense_8/BiasAdd/ReadVariableOp2>
dense_8/MatMul/ReadVariableOpdense_8/MatMul/ReadVariableOp:W S
/
_output_shapes
:���������HX
 
_user_specified_nameinputs
�
�
O__inference_CBlock_2_BatchNorm_layer_call_and_return_conditional_losses_1690475

inputs%
readvariableop_resource:1'
readvariableop_1_resource:16
(fusedbatchnormv3_readvariableop_resource:18
*fusedbatchnormv3_readvariableop_1_resource:1
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:1*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:1*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:1*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:1*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������1:1:1:1:1:*
epsilon%o�:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������1�
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������1: : : : 2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:i e
A
_output_shapes/
-:+���������������������������1
 
_user_specified_nameinputs
�
�
O__inference_CBlock_2_BatchNorm_layer_call_and_return_conditional_losses_1688866

inputs%
readvariableop_resource:1'
readvariableop_1_resource:16
(fusedbatchnormv3_readvariableop_resource:18
*fusedbatchnormv3_readvariableop_1_resource:1
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:1*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:1*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:1*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:1*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������1:1:1:1:1:*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������1�
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������1: : : : 2$
AssignNewValue_1AssignNewValue_12 
AssignNewValueAssignNewValue2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:i e
A
_output_shapes/
-:+���������������������������1
 
_user_specified_nameinputs
�
�
F__inference_jojo_n8_r72x88_a0.7_b4_g2.2_strides2_layer_call_fn_1690024

inputs!
unknown:
	unknown_0:#
	unknown_1:#
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:#
	unknown_8:#
	unknown_9:1

unknown_10:1

unknown_11:1

unknown_12:1

unknown_13:1

unknown_14:1$

unknown_15:1$

unknown_16:1J

unknown_17:J

unknown_18:J

unknown_19:J

unknown_20:J

unknown_21:J$

unknown_22:J$

unknown_23:Jc

unknown_24:c

unknown_25:c

unknown_26:c

unknown_27:c

unknown_28:c

unknown_29:c

unknown_30:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30*,
Tin%
#2!*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*B
_read_only_resource_inputs$
" 	
 *0
config_proto 

CPU

GPU2*0J 8� *j
feRc
a__inference_jojo_n8_r72x88_a0.7_b4_g2.2_strides2_layer_call_and_return_conditional_losses_1689560o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*n
_input_shapes]
[:���������HX: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������HX
 
_user_specified_nameinputs
�	
�
4__inference_CBlock_1_SepConv2D_layer_call_fn_1690300

inputs!
unknown:#
	unknown_0:
	unknown_1:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *X
fSRQ
O__inference_CBlock_1_SepConv2D_layer_call_and_return_conditional_losses_1688743�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:+���������������������������: : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�	
�
4__inference_CBlock_3_BatchNorm_layer_call_fn_1690524

inputs
unknown:J
	unknown_0:J
	unknown_1:J
	unknown_2:J
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������J*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *X
fSRQ
O__inference_CBlock_3_BatchNorm_layer_call_and_return_conditional_losses_1688958�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������J`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������J: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������J
 
_user_specified_nameinputs
�
K
/__inference_CBlock_3_ReLu_layer_call_fn_1690578

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������J* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_CBlock_3_ReLu_layer_call_and_return_conditional_losses_1689195h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������J"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������J:W S
/
_output_shapes
:���������J
 
_user_specified_nameinputs
�
�
O__inference_CBlock_4_BatchNorm_layer_call_and_return_conditional_losses_1689050

inputs%
readvariableop_resource:c'
readvariableop_1_resource:c6
(fusedbatchnormv3_readvariableop_resource:c8
*fusedbatchnormv3_readvariableop_1_resource:c
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:c*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:c*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:c*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:c*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������c:c:c:c:c:*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������c�
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������c: : : : 2$
AssignNewValue_1AssignNewValue_12 
AssignNewValueAssignNewValue2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:i e
A
_output_shapes/
-:+���������������������������c
 
_user_specified_nameinputs
�Q
�
a__inference_jojo_n8_r72x88_a0.7_b4_g2.2_strides2_layer_call_and_return_conditional_losses_1689239
input_layer'
conv0_1689123:
conv0_1689125:4
cblock_1_sepconv2d_1689128:4
cblock_1_sepconv2d_1689130:(
cblock_1_sepconv2d_1689132:(
cblock_1_batchnorm_1689135:(
cblock_1_batchnorm_1689137:(
cblock_1_batchnorm_1689139:(
cblock_1_batchnorm_1689141:4
cblock_2_sepconv2d_1689151:4
cblock_2_sepconv2d_1689153:1(
cblock_2_sepconv2d_1689155:1(
cblock_2_batchnorm_1689158:1(
cblock_2_batchnorm_1689160:1(
cblock_2_batchnorm_1689162:1(
cblock_2_batchnorm_1689164:14
cblock_3_sepconv2d_1689174:14
cblock_3_sepconv2d_1689176:1J(
cblock_3_sepconv2d_1689178:J(
cblock_3_batchnorm_1689181:J(
cblock_3_batchnorm_1689183:J(
cblock_3_batchnorm_1689185:J(
cblock_3_batchnorm_1689187:J4
cblock_4_sepconv2d_1689197:J4
cblock_4_sepconv2d_1689199:Jc(
cblock_4_sepconv2d_1689201:c(
cblock_4_batchnorm_1689204:c(
cblock_4_batchnorm_1689206:c(
cblock_4_batchnorm_1689208:c(
cblock_4_batchnorm_1689210:c!
dense_8_1689233:c
dense_8_1689235:
identity��*CBlock_1_BatchNorm/StatefulPartitionedCall�*CBlock_1_SepConv2D/StatefulPartitionedCall�*CBlock_2_BatchNorm/StatefulPartitionedCall�*CBlock_2_SepConv2D/StatefulPartitionedCall�*CBlock_3_BatchNorm/StatefulPartitionedCall�*CBlock_3_SepConv2D/StatefulPartitionedCall�*CBlock_4_BatchNorm/StatefulPartitionedCall�*CBlock_4_SepConv2D/StatefulPartitionedCall�Conv0/StatefulPartitionedCall�dense_8/StatefulPartitionedCall�
Conv0/StatefulPartitionedCallStatefulPartitionedCallinput_layerconv0_1689123conv0_1689125*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������$,*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_Conv0_layer_call_and_return_conditional_losses_1689122�
*CBlock_1_SepConv2D/StatefulPartitionedCallStatefulPartitionedCall&Conv0/StatefulPartitionedCall:output:0cblock_1_sepconv2d_1689128cblock_1_sepconv2d_1689130cblock_1_sepconv2d_1689132*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *X
fSRQ
O__inference_CBlock_1_SepConv2D_layer_call_and_return_conditional_losses_1688743�
*CBlock_1_BatchNorm/StatefulPartitionedCallStatefulPartitionedCall3CBlock_1_SepConv2D/StatefulPartitionedCall:output:0cblock_1_batchnorm_1689135cblock_1_batchnorm_1689137cblock_1_batchnorm_1689139cblock_1_batchnorm_1689141*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *X
fSRQ
O__inference_CBlock_1_BatchNorm_layer_call_and_return_conditional_losses_1688774�
CBlock_1_ReLu/PartitionedCallPartitionedCall3CBlock_1_BatchNorm/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_CBlock_1_ReLu_layer_call_and_return_conditional_losses_1689149�
*CBlock_2_SepConv2D/StatefulPartitionedCallStatefulPartitionedCall&CBlock_1_ReLu/PartitionedCall:output:0cblock_2_sepconv2d_1689151cblock_2_sepconv2d_1689153cblock_2_sepconv2d_1689155*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������	1*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *X
fSRQ
O__inference_CBlock_2_SepConv2D_layer_call_and_return_conditional_losses_1688835�
*CBlock_2_BatchNorm/StatefulPartitionedCallStatefulPartitionedCall3CBlock_2_SepConv2D/StatefulPartitionedCall:output:0cblock_2_batchnorm_1689158cblock_2_batchnorm_1689160cblock_2_batchnorm_1689162cblock_2_batchnorm_1689164*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������	1*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *X
fSRQ
O__inference_CBlock_2_BatchNorm_layer_call_and_return_conditional_losses_1688866�
CBlock_2_ReLu/PartitionedCallPartitionedCall3CBlock_2_BatchNorm/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������	1* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_CBlock_2_ReLu_layer_call_and_return_conditional_losses_1689172�
*CBlock_3_SepConv2D/StatefulPartitionedCallStatefulPartitionedCall&CBlock_2_ReLu/PartitionedCall:output:0cblock_3_sepconv2d_1689174cblock_3_sepconv2d_1689176cblock_3_sepconv2d_1689178*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������J*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *X
fSRQ
O__inference_CBlock_3_SepConv2D_layer_call_and_return_conditional_losses_1688927�
*CBlock_3_BatchNorm/StatefulPartitionedCallStatefulPartitionedCall3CBlock_3_SepConv2D/StatefulPartitionedCall:output:0cblock_3_batchnorm_1689181cblock_3_batchnorm_1689183cblock_3_batchnorm_1689185cblock_3_batchnorm_1689187*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������J*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *X
fSRQ
O__inference_CBlock_3_BatchNorm_layer_call_and_return_conditional_losses_1688958�
CBlock_3_ReLu/PartitionedCallPartitionedCall3CBlock_3_BatchNorm/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������J* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_CBlock_3_ReLu_layer_call_and_return_conditional_losses_1689195�
*CBlock_4_SepConv2D/StatefulPartitionedCallStatefulPartitionedCall&CBlock_3_ReLu/PartitionedCall:output:0cblock_4_sepconv2d_1689197cblock_4_sepconv2d_1689199cblock_4_sepconv2d_1689201*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������c*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *X
fSRQ
O__inference_CBlock_4_SepConv2D_layer_call_and_return_conditional_losses_1689019�
*CBlock_4_BatchNorm/StatefulPartitionedCallStatefulPartitionedCall3CBlock_4_SepConv2D/StatefulPartitionedCall:output:0cblock_4_batchnorm_1689204cblock_4_batchnorm_1689206cblock_4_batchnorm_1689208cblock_4_batchnorm_1689210*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������c*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *X
fSRQ
O__inference_CBlock_4_BatchNorm_layer_call_and_return_conditional_losses_1689050�
CBlock_4_ReLu/PartitionedCallPartitionedCall3CBlock_4_BatchNorm/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������c* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_CBlock_4_ReLu_layer_call_and_return_conditional_losses_1689218�
*global_average_pooling2d_8/PartitionedCallPartitionedCall&CBlock_4_ReLu/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������c* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *`
f[RY
W__inference_global_average_pooling2d_8_layer_call_and_return_conditional_losses_1689102�
dense_8/StatefulPartitionedCallStatefulPartitionedCall3global_average_pooling2d_8/PartitionedCall:output:0dense_8_1689233dense_8_1689235*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_dense_8_layer_call_and_return_conditional_losses_1689232w
IdentityIdentity(dense_8/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp+^CBlock_1_BatchNorm/StatefulPartitionedCall+^CBlock_1_SepConv2D/StatefulPartitionedCall+^CBlock_2_BatchNorm/StatefulPartitionedCall+^CBlock_2_SepConv2D/StatefulPartitionedCall+^CBlock_3_BatchNorm/StatefulPartitionedCall+^CBlock_3_SepConv2D/StatefulPartitionedCall+^CBlock_4_BatchNorm/StatefulPartitionedCall+^CBlock_4_SepConv2D/StatefulPartitionedCall^Conv0/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*n
_input_shapes]
[:���������HX: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2X
*CBlock_1_BatchNorm/StatefulPartitionedCall*CBlock_1_BatchNorm/StatefulPartitionedCall2X
*CBlock_1_SepConv2D/StatefulPartitionedCall*CBlock_1_SepConv2D/StatefulPartitionedCall2X
*CBlock_2_BatchNorm/StatefulPartitionedCall*CBlock_2_BatchNorm/StatefulPartitionedCall2X
*CBlock_2_SepConv2D/StatefulPartitionedCall*CBlock_2_SepConv2D/StatefulPartitionedCall2X
*CBlock_3_BatchNorm/StatefulPartitionedCall*CBlock_3_BatchNorm/StatefulPartitionedCall2X
*CBlock_3_SepConv2D/StatefulPartitionedCall*CBlock_3_SepConv2D/StatefulPartitionedCall2X
*CBlock_4_BatchNorm/StatefulPartitionedCall*CBlock_4_BatchNorm/StatefulPartitionedCall2X
*CBlock_4_SepConv2D/StatefulPartitionedCall*CBlock_4_SepConv2D/StatefulPartitionedCall2>
Conv0/StatefulPartitionedCallConv0/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall:\ X
/
_output_shapes
:���������HX
%
_user_specified_nameinput_layer
�
�
O__inference_CBlock_4_BatchNorm_layer_call_and_return_conditional_losses_1690671

inputs%
readvariableop_resource:c'
readvariableop_1_resource:c6
(fusedbatchnormv3_readvariableop_resource:c8
*fusedbatchnormv3_readvariableop_1_resource:c
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:c*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:c*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:c*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:c*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������c:c:c:c:c:*
epsilon%o�:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������c�
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������c: : : : 2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:i e
A
_output_shapes/
-:+���������������������������c
 
_user_specified_nameinputs
�	
�
4__inference_CBlock_2_SepConv2D_layer_call_fn_1690398

inputs!
unknown:#
	unknown_0:1
	unknown_1:1
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������1*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *X
fSRQ
O__inference_CBlock_2_SepConv2D_layer_call_and_return_conditional_losses_1688835�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������1`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:+���������������������������: : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
�
O__inference_CBlock_1_BatchNorm_layer_call_and_return_conditional_losses_1688792

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������:::::*
epsilon%o�:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+����������������������������
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������: : : : 2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
f
J__inference_CBlock_2_ReLu_layer_call_and_return_conditional_losses_1689172

inputs
identityP
Relu6Relu6inputs*
T0*/
_output_shapes
:���������	1c
IdentityIdentityRelu6:activations:0*
T0*/
_output_shapes
:���������	1"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������	1:W S
/
_output_shapes
:���������	1
 
_user_specified_nameinputs
�
�
)__inference_dense_8_layer_call_fn_1690701

inputs
unknown:c
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_dense_8_layer_call_and_return_conditional_losses_1689232o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������c: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������c
 
_user_specified_nameinputs
�
f
J__inference_CBlock_3_ReLu_layer_call_and_return_conditional_losses_1689195

inputs
identityP
Relu6Relu6inputs*
T0*/
_output_shapes
:���������Jc
IdentityIdentityRelu6:activations:0*
T0*/
_output_shapes
:���������J"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������J:W S
/
_output_shapes
:���������J
 
_user_specified_nameinputs
�Q
�
a__inference_jojo_n8_r72x88_a0.7_b4_g2.2_strides2_layer_call_and_return_conditional_losses_1689560

inputs'
conv0_1689480:
conv0_1689482:4
cblock_1_sepconv2d_1689485:4
cblock_1_sepconv2d_1689487:(
cblock_1_sepconv2d_1689489:(
cblock_1_batchnorm_1689492:(
cblock_1_batchnorm_1689494:(
cblock_1_batchnorm_1689496:(
cblock_1_batchnorm_1689498:4
cblock_2_sepconv2d_1689502:4
cblock_2_sepconv2d_1689504:1(
cblock_2_sepconv2d_1689506:1(
cblock_2_batchnorm_1689509:1(
cblock_2_batchnorm_1689511:1(
cblock_2_batchnorm_1689513:1(
cblock_2_batchnorm_1689515:14
cblock_3_sepconv2d_1689519:14
cblock_3_sepconv2d_1689521:1J(
cblock_3_sepconv2d_1689523:J(
cblock_3_batchnorm_1689526:J(
cblock_3_batchnorm_1689528:J(
cblock_3_batchnorm_1689530:J(
cblock_3_batchnorm_1689532:J4
cblock_4_sepconv2d_1689536:J4
cblock_4_sepconv2d_1689538:Jc(
cblock_4_sepconv2d_1689540:c(
cblock_4_batchnorm_1689543:c(
cblock_4_batchnorm_1689545:c(
cblock_4_batchnorm_1689547:c(
cblock_4_batchnorm_1689549:c!
dense_8_1689554:c
dense_8_1689556:
identity��*CBlock_1_BatchNorm/StatefulPartitionedCall�*CBlock_1_SepConv2D/StatefulPartitionedCall�*CBlock_2_BatchNorm/StatefulPartitionedCall�*CBlock_2_SepConv2D/StatefulPartitionedCall�*CBlock_3_BatchNorm/StatefulPartitionedCall�*CBlock_3_SepConv2D/StatefulPartitionedCall�*CBlock_4_BatchNorm/StatefulPartitionedCall�*CBlock_4_SepConv2D/StatefulPartitionedCall�Conv0/StatefulPartitionedCall�dense_8/StatefulPartitionedCall�
Conv0/StatefulPartitionedCallStatefulPartitionedCallinputsconv0_1689480conv0_1689482*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������$,*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_Conv0_layer_call_and_return_conditional_losses_1689122�
*CBlock_1_SepConv2D/StatefulPartitionedCallStatefulPartitionedCall&Conv0/StatefulPartitionedCall:output:0cblock_1_sepconv2d_1689485cblock_1_sepconv2d_1689487cblock_1_sepconv2d_1689489*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *X
fSRQ
O__inference_CBlock_1_SepConv2D_layer_call_and_return_conditional_losses_1688743�
*CBlock_1_BatchNorm/StatefulPartitionedCallStatefulPartitionedCall3CBlock_1_SepConv2D/StatefulPartitionedCall:output:0cblock_1_batchnorm_1689492cblock_1_batchnorm_1689494cblock_1_batchnorm_1689496cblock_1_batchnorm_1689498*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *X
fSRQ
O__inference_CBlock_1_BatchNorm_layer_call_and_return_conditional_losses_1688792�
CBlock_1_ReLu/PartitionedCallPartitionedCall3CBlock_1_BatchNorm/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_CBlock_1_ReLu_layer_call_and_return_conditional_losses_1689149�
*CBlock_2_SepConv2D/StatefulPartitionedCallStatefulPartitionedCall&CBlock_1_ReLu/PartitionedCall:output:0cblock_2_sepconv2d_1689502cblock_2_sepconv2d_1689504cblock_2_sepconv2d_1689506*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������	1*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *X
fSRQ
O__inference_CBlock_2_SepConv2D_layer_call_and_return_conditional_losses_1688835�
*CBlock_2_BatchNorm/StatefulPartitionedCallStatefulPartitionedCall3CBlock_2_SepConv2D/StatefulPartitionedCall:output:0cblock_2_batchnorm_1689509cblock_2_batchnorm_1689511cblock_2_batchnorm_1689513cblock_2_batchnorm_1689515*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������	1*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *X
fSRQ
O__inference_CBlock_2_BatchNorm_layer_call_and_return_conditional_losses_1688884�
CBlock_2_ReLu/PartitionedCallPartitionedCall3CBlock_2_BatchNorm/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������	1* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_CBlock_2_ReLu_layer_call_and_return_conditional_losses_1689172�
*CBlock_3_SepConv2D/StatefulPartitionedCallStatefulPartitionedCall&CBlock_2_ReLu/PartitionedCall:output:0cblock_3_sepconv2d_1689519cblock_3_sepconv2d_1689521cblock_3_sepconv2d_1689523*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������J*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *X
fSRQ
O__inference_CBlock_3_SepConv2D_layer_call_and_return_conditional_losses_1688927�
*CBlock_3_BatchNorm/StatefulPartitionedCallStatefulPartitionedCall3CBlock_3_SepConv2D/StatefulPartitionedCall:output:0cblock_3_batchnorm_1689526cblock_3_batchnorm_1689528cblock_3_batchnorm_1689530cblock_3_batchnorm_1689532*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������J*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *X
fSRQ
O__inference_CBlock_3_BatchNorm_layer_call_and_return_conditional_losses_1688976�
CBlock_3_ReLu/PartitionedCallPartitionedCall3CBlock_3_BatchNorm/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������J* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_CBlock_3_ReLu_layer_call_and_return_conditional_losses_1689195�
*CBlock_4_SepConv2D/StatefulPartitionedCallStatefulPartitionedCall&CBlock_3_ReLu/PartitionedCall:output:0cblock_4_sepconv2d_1689536cblock_4_sepconv2d_1689538cblock_4_sepconv2d_1689540*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������c*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *X
fSRQ
O__inference_CBlock_4_SepConv2D_layer_call_and_return_conditional_losses_1689019�
*CBlock_4_BatchNorm/StatefulPartitionedCallStatefulPartitionedCall3CBlock_4_SepConv2D/StatefulPartitionedCall:output:0cblock_4_batchnorm_1689543cblock_4_batchnorm_1689545cblock_4_batchnorm_1689547cblock_4_batchnorm_1689549*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������c*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *X
fSRQ
O__inference_CBlock_4_BatchNorm_layer_call_and_return_conditional_losses_1689068�
CBlock_4_ReLu/PartitionedCallPartitionedCall3CBlock_4_BatchNorm/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������c* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_CBlock_4_ReLu_layer_call_and_return_conditional_losses_1689218�
*global_average_pooling2d_8/PartitionedCallPartitionedCall&CBlock_4_ReLu/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������c* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *`
f[RY
W__inference_global_average_pooling2d_8_layer_call_and_return_conditional_losses_1689102�
dense_8/StatefulPartitionedCallStatefulPartitionedCall3global_average_pooling2d_8/PartitionedCall:output:0dense_8_1689554dense_8_1689556*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_dense_8_layer_call_and_return_conditional_losses_1689232w
IdentityIdentity(dense_8/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp+^CBlock_1_BatchNorm/StatefulPartitionedCall+^CBlock_1_SepConv2D/StatefulPartitionedCall+^CBlock_2_BatchNorm/StatefulPartitionedCall+^CBlock_2_SepConv2D/StatefulPartitionedCall+^CBlock_3_BatchNorm/StatefulPartitionedCall+^CBlock_3_SepConv2D/StatefulPartitionedCall+^CBlock_4_BatchNorm/StatefulPartitionedCall+^CBlock_4_SepConv2D/StatefulPartitionedCall^Conv0/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*n
_input_shapes]
[:���������HX: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2X
*CBlock_1_BatchNorm/StatefulPartitionedCall*CBlock_1_BatchNorm/StatefulPartitionedCall2X
*CBlock_1_SepConv2D/StatefulPartitionedCall*CBlock_1_SepConv2D/StatefulPartitionedCall2X
*CBlock_2_BatchNorm/StatefulPartitionedCall*CBlock_2_BatchNorm/StatefulPartitionedCall2X
*CBlock_2_SepConv2D/StatefulPartitionedCall*CBlock_2_SepConv2D/StatefulPartitionedCall2X
*CBlock_3_BatchNorm/StatefulPartitionedCall*CBlock_3_BatchNorm/StatefulPartitionedCall2X
*CBlock_3_SepConv2D/StatefulPartitionedCall*CBlock_3_SepConv2D/StatefulPartitionedCall2X
*CBlock_4_BatchNorm/StatefulPartitionedCall*CBlock_4_BatchNorm/StatefulPartitionedCall2X
*CBlock_4_SepConv2D/StatefulPartitionedCall*CBlock_4_SepConv2D/StatefulPartitionedCall2>
Conv0/StatefulPartitionedCallConv0/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall:W S
/
_output_shapes
:���������HX
 
_user_specified_nameinputs
�
�
F__inference_jojo_n8_r72x88_a0.7_b4_g2.2_strides2_layer_call_fn_1689627
input_layer!
unknown:
	unknown_0:#
	unknown_1:#
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:#
	unknown_8:#
	unknown_9:1

unknown_10:1

unknown_11:1

unknown_12:1

unknown_13:1

unknown_14:1$

unknown_15:1$

unknown_16:1J

unknown_17:J

unknown_18:J

unknown_19:J

unknown_20:J

unknown_21:J$

unknown_22:J$

unknown_23:Jc

unknown_24:c

unknown_25:c

unknown_26:c

unknown_27:c

unknown_28:c

unknown_29:c

unknown_30:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_layerunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30*,
Tin%
#2!*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*B
_read_only_resource_inputs$
" 	
 *0
config_proto 

CPU

GPU2*0J 8� *j
feRc
a__inference_jojo_n8_r72x88_a0.7_b4_g2.2_strides2_layer_call_and_return_conditional_losses_1689560o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*n
_input_shapes]
[:���������HX: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
/
_output_shapes
:���������HX
%
_user_specified_nameinput_layer
�
�
O__inference_CBlock_1_BatchNorm_layer_call_and_return_conditional_losses_1690377

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������:::::*
epsilon%o�:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+����������������������������
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������: : : : 2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
�
O__inference_CBlock_1_SepConv2D_layer_call_and_return_conditional_losses_1688743

inputsB
(separable_conv2d_readvariableop_resource:D
*separable_conv2d_readvariableop_1_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�separable_conv2d/ReadVariableOp�!separable_conv2d/ReadVariableOp_1�
separable_conv2d/ReadVariableOpReadVariableOp(separable_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
!separable_conv2d/ReadVariableOp_1ReadVariableOp*separable_conv2d_readvariableop_1_resource*&
_output_shapes
:*
dtype0o
separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            o
separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      �
separable_conv2d/depthwiseDepthwiseConv2dNativeinputs'separable_conv2d/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������*
paddingSAME*
strides
�
separable_conv2dConv2D#separable_conv2d/depthwise:output:0)separable_conv2d/ReadVariableOp_1:value:0*
T0*A
_output_shapes/
-:+���������������������������*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
BiasAddBiasAddseparable_conv2d:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������y
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+����������������������������
NoOpNoOp^BiasAdd/ReadVariableOp ^separable_conv2d/ReadVariableOp"^separable_conv2d/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:+���������������������������: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2F
!separable_conv2d/ReadVariableOp_1!separable_conv2d/ReadVariableOp_12B
separable_conv2d/ReadVariableOpseparable_conv2d/ReadVariableOp:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
f
J__inference_CBlock_4_ReLu_layer_call_and_return_conditional_losses_1689218

inputs
identityP
Relu6Relu6inputs*
T0*/
_output_shapes
:���������cc
IdentityIdentityRelu6:activations:0*
T0*/
_output_shapes
:���������c"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������c:W S
/
_output_shapes
:���������c
 
_user_specified_nameinputs
�	
�
4__inference_CBlock_2_BatchNorm_layer_call_fn_1690439

inputs
unknown:1
	unknown_0:1
	unknown_1:1
	unknown_2:1
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������1*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *X
fSRQ
O__inference_CBlock_2_BatchNorm_layer_call_and_return_conditional_losses_1688884�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������1`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������1: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������1
 
_user_specified_nameinputs
�
�
O__inference_CBlock_1_BatchNorm_layer_call_and_return_conditional_losses_1688774

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������:::::*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+����������������������������
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������: : : : 2$
AssignNewValue_1AssignNewValue_12 
AssignNewValueAssignNewValue2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
�
O__inference_CBlock_3_SepConv2D_layer_call_and_return_conditional_losses_1690511

inputsB
(separable_conv2d_readvariableop_resource:1D
*separable_conv2d_readvariableop_1_resource:1J-
biasadd_readvariableop_resource:J
identity��BiasAdd/ReadVariableOp�separable_conv2d/ReadVariableOp�!separable_conv2d/ReadVariableOp_1�
separable_conv2d/ReadVariableOpReadVariableOp(separable_conv2d_readvariableop_resource*&
_output_shapes
:1*
dtype0�
!separable_conv2d/ReadVariableOp_1ReadVariableOp*separable_conv2d_readvariableop_1_resource*&
_output_shapes
:1J*
dtype0o
separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      1      o
separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      �
separable_conv2d/depthwiseDepthwiseConv2dNativeinputs'separable_conv2d/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������1*
paddingSAME*
strides
�
separable_conv2dConv2D#separable_conv2d/depthwise:output:0)separable_conv2d/ReadVariableOp_1:value:0*
T0*A
_output_shapes/
-:+���������������������������J*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:J*
dtype0�
BiasAddBiasAddseparable_conv2d:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������Jy
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������J�
NoOpNoOp^BiasAdd/ReadVariableOp ^separable_conv2d/ReadVariableOp"^separable_conv2d/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:+���������������������������1: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2F
!separable_conv2d/ReadVariableOp_1!separable_conv2d/ReadVariableOp_12B
separable_conv2d/ReadVariableOpseparable_conv2d/ReadVariableOp:i e
A
_output_shapes/
-:+���������������������������1
 
_user_specified_nameinputs
�Q
�
a__inference_jojo_n8_r72x88_a0.7_b4_g2.2_strides2_layer_call_and_return_conditional_losses_1689408

inputs'
conv0_1689328:
conv0_1689330:4
cblock_1_sepconv2d_1689333:4
cblock_1_sepconv2d_1689335:(
cblock_1_sepconv2d_1689337:(
cblock_1_batchnorm_1689340:(
cblock_1_batchnorm_1689342:(
cblock_1_batchnorm_1689344:(
cblock_1_batchnorm_1689346:4
cblock_2_sepconv2d_1689350:4
cblock_2_sepconv2d_1689352:1(
cblock_2_sepconv2d_1689354:1(
cblock_2_batchnorm_1689357:1(
cblock_2_batchnorm_1689359:1(
cblock_2_batchnorm_1689361:1(
cblock_2_batchnorm_1689363:14
cblock_3_sepconv2d_1689367:14
cblock_3_sepconv2d_1689369:1J(
cblock_3_sepconv2d_1689371:J(
cblock_3_batchnorm_1689374:J(
cblock_3_batchnorm_1689376:J(
cblock_3_batchnorm_1689378:J(
cblock_3_batchnorm_1689380:J4
cblock_4_sepconv2d_1689384:J4
cblock_4_sepconv2d_1689386:Jc(
cblock_4_sepconv2d_1689388:c(
cblock_4_batchnorm_1689391:c(
cblock_4_batchnorm_1689393:c(
cblock_4_batchnorm_1689395:c(
cblock_4_batchnorm_1689397:c!
dense_8_1689402:c
dense_8_1689404:
identity��*CBlock_1_BatchNorm/StatefulPartitionedCall�*CBlock_1_SepConv2D/StatefulPartitionedCall�*CBlock_2_BatchNorm/StatefulPartitionedCall�*CBlock_2_SepConv2D/StatefulPartitionedCall�*CBlock_3_BatchNorm/StatefulPartitionedCall�*CBlock_3_SepConv2D/StatefulPartitionedCall�*CBlock_4_BatchNorm/StatefulPartitionedCall�*CBlock_4_SepConv2D/StatefulPartitionedCall�Conv0/StatefulPartitionedCall�dense_8/StatefulPartitionedCall�
Conv0/StatefulPartitionedCallStatefulPartitionedCallinputsconv0_1689328conv0_1689330*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������$,*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_Conv0_layer_call_and_return_conditional_losses_1689122�
*CBlock_1_SepConv2D/StatefulPartitionedCallStatefulPartitionedCall&Conv0/StatefulPartitionedCall:output:0cblock_1_sepconv2d_1689333cblock_1_sepconv2d_1689335cblock_1_sepconv2d_1689337*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *X
fSRQ
O__inference_CBlock_1_SepConv2D_layer_call_and_return_conditional_losses_1688743�
*CBlock_1_BatchNorm/StatefulPartitionedCallStatefulPartitionedCall3CBlock_1_SepConv2D/StatefulPartitionedCall:output:0cblock_1_batchnorm_1689340cblock_1_batchnorm_1689342cblock_1_batchnorm_1689344cblock_1_batchnorm_1689346*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *X
fSRQ
O__inference_CBlock_1_BatchNorm_layer_call_and_return_conditional_losses_1688774�
CBlock_1_ReLu/PartitionedCallPartitionedCall3CBlock_1_BatchNorm/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_CBlock_1_ReLu_layer_call_and_return_conditional_losses_1689149�
*CBlock_2_SepConv2D/StatefulPartitionedCallStatefulPartitionedCall&CBlock_1_ReLu/PartitionedCall:output:0cblock_2_sepconv2d_1689350cblock_2_sepconv2d_1689352cblock_2_sepconv2d_1689354*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������	1*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *X
fSRQ
O__inference_CBlock_2_SepConv2D_layer_call_and_return_conditional_losses_1688835�
*CBlock_2_BatchNorm/StatefulPartitionedCallStatefulPartitionedCall3CBlock_2_SepConv2D/StatefulPartitionedCall:output:0cblock_2_batchnorm_1689357cblock_2_batchnorm_1689359cblock_2_batchnorm_1689361cblock_2_batchnorm_1689363*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������	1*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *X
fSRQ
O__inference_CBlock_2_BatchNorm_layer_call_and_return_conditional_losses_1688866�
CBlock_2_ReLu/PartitionedCallPartitionedCall3CBlock_2_BatchNorm/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������	1* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_CBlock_2_ReLu_layer_call_and_return_conditional_losses_1689172�
*CBlock_3_SepConv2D/StatefulPartitionedCallStatefulPartitionedCall&CBlock_2_ReLu/PartitionedCall:output:0cblock_3_sepconv2d_1689367cblock_3_sepconv2d_1689369cblock_3_sepconv2d_1689371*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������J*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *X
fSRQ
O__inference_CBlock_3_SepConv2D_layer_call_and_return_conditional_losses_1688927�
*CBlock_3_BatchNorm/StatefulPartitionedCallStatefulPartitionedCall3CBlock_3_SepConv2D/StatefulPartitionedCall:output:0cblock_3_batchnorm_1689374cblock_3_batchnorm_1689376cblock_3_batchnorm_1689378cblock_3_batchnorm_1689380*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������J*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *X
fSRQ
O__inference_CBlock_3_BatchNorm_layer_call_and_return_conditional_losses_1688958�
CBlock_3_ReLu/PartitionedCallPartitionedCall3CBlock_3_BatchNorm/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������J* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_CBlock_3_ReLu_layer_call_and_return_conditional_losses_1689195�
*CBlock_4_SepConv2D/StatefulPartitionedCallStatefulPartitionedCall&CBlock_3_ReLu/PartitionedCall:output:0cblock_4_sepconv2d_1689384cblock_4_sepconv2d_1689386cblock_4_sepconv2d_1689388*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������c*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *X
fSRQ
O__inference_CBlock_4_SepConv2D_layer_call_and_return_conditional_losses_1689019�
*CBlock_4_BatchNorm/StatefulPartitionedCallStatefulPartitionedCall3CBlock_4_SepConv2D/StatefulPartitionedCall:output:0cblock_4_batchnorm_1689391cblock_4_batchnorm_1689393cblock_4_batchnorm_1689395cblock_4_batchnorm_1689397*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������c*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *X
fSRQ
O__inference_CBlock_4_BatchNorm_layer_call_and_return_conditional_losses_1689050�
CBlock_4_ReLu/PartitionedCallPartitionedCall3CBlock_4_BatchNorm/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������c* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_CBlock_4_ReLu_layer_call_and_return_conditional_losses_1689218�
*global_average_pooling2d_8/PartitionedCallPartitionedCall&CBlock_4_ReLu/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������c* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *`
f[RY
W__inference_global_average_pooling2d_8_layer_call_and_return_conditional_losses_1689102�
dense_8/StatefulPartitionedCallStatefulPartitionedCall3global_average_pooling2d_8/PartitionedCall:output:0dense_8_1689402dense_8_1689404*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_dense_8_layer_call_and_return_conditional_losses_1689232w
IdentityIdentity(dense_8/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp+^CBlock_1_BatchNorm/StatefulPartitionedCall+^CBlock_1_SepConv2D/StatefulPartitionedCall+^CBlock_2_BatchNorm/StatefulPartitionedCall+^CBlock_2_SepConv2D/StatefulPartitionedCall+^CBlock_3_BatchNorm/StatefulPartitionedCall+^CBlock_3_SepConv2D/StatefulPartitionedCall+^CBlock_4_BatchNorm/StatefulPartitionedCall+^CBlock_4_SepConv2D/StatefulPartitionedCall^Conv0/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*n
_input_shapes]
[:���������HX: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2X
*CBlock_1_BatchNorm/StatefulPartitionedCall*CBlock_1_BatchNorm/StatefulPartitionedCall2X
*CBlock_1_SepConv2D/StatefulPartitionedCall*CBlock_1_SepConv2D/StatefulPartitionedCall2X
*CBlock_2_BatchNorm/StatefulPartitionedCall*CBlock_2_BatchNorm/StatefulPartitionedCall2X
*CBlock_2_SepConv2D/StatefulPartitionedCall*CBlock_2_SepConv2D/StatefulPartitionedCall2X
*CBlock_3_BatchNorm/StatefulPartitionedCall*CBlock_3_BatchNorm/StatefulPartitionedCall2X
*CBlock_3_SepConv2D/StatefulPartitionedCall*CBlock_3_SepConv2D/StatefulPartitionedCall2X
*CBlock_4_BatchNorm/StatefulPartitionedCall*CBlock_4_BatchNorm/StatefulPartitionedCall2X
*CBlock_4_SepConv2D/StatefulPartitionedCall*CBlock_4_SepConv2D/StatefulPartitionedCall2>
Conv0/StatefulPartitionedCallConv0/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall:W S
/
_output_shapes
:���������HX
 
_user_specified_nameinputs
�
s
W__inference_global_average_pooling2d_8_layer_call_and_return_conditional_losses_1690692

inputs
identityg
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      p
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:������������������^
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
�
O__inference_CBlock_4_SepConv2D_layer_call_and_return_conditional_losses_1690609

inputsB
(separable_conv2d_readvariableop_resource:JD
*separable_conv2d_readvariableop_1_resource:Jc-
biasadd_readvariableop_resource:c
identity��BiasAdd/ReadVariableOp�separable_conv2d/ReadVariableOp�!separable_conv2d/ReadVariableOp_1�
separable_conv2d/ReadVariableOpReadVariableOp(separable_conv2d_readvariableop_resource*&
_output_shapes
:J*
dtype0�
!separable_conv2d/ReadVariableOp_1ReadVariableOp*separable_conv2d_readvariableop_1_resource*&
_output_shapes
:Jc*
dtype0o
separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      J      o
separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      �
separable_conv2d/depthwiseDepthwiseConv2dNativeinputs'separable_conv2d/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������J*
paddingSAME*
strides
�
separable_conv2dConv2D#separable_conv2d/depthwise:output:0)separable_conv2d/ReadVariableOp_1:value:0*
T0*A
_output_shapes/
-:+���������������������������c*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:c*
dtype0�
BiasAddBiasAddseparable_conv2d:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������cy
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������c�
NoOpNoOp^BiasAdd/ReadVariableOp ^separable_conv2d/ReadVariableOp"^separable_conv2d/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:+���������������������������J: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2F
!separable_conv2d/ReadVariableOp_1!separable_conv2d/ReadVariableOp_12B
separable_conv2d/ReadVariableOpseparable_conv2d/ReadVariableOp:i e
A
_output_shapes/
-:+���������������������������J
 
_user_specified_nameinputs
�
�
O__inference_CBlock_4_SepConv2D_layer_call_and_return_conditional_losses_1689019

inputsB
(separable_conv2d_readvariableop_resource:JD
*separable_conv2d_readvariableop_1_resource:Jc-
biasadd_readvariableop_resource:c
identity��BiasAdd/ReadVariableOp�separable_conv2d/ReadVariableOp�!separable_conv2d/ReadVariableOp_1�
separable_conv2d/ReadVariableOpReadVariableOp(separable_conv2d_readvariableop_resource*&
_output_shapes
:J*
dtype0�
!separable_conv2d/ReadVariableOp_1ReadVariableOp*separable_conv2d_readvariableop_1_resource*&
_output_shapes
:Jc*
dtype0o
separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      J      o
separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      �
separable_conv2d/depthwiseDepthwiseConv2dNativeinputs'separable_conv2d/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������J*
paddingSAME*
strides
�
separable_conv2dConv2D#separable_conv2d/depthwise:output:0)separable_conv2d/ReadVariableOp_1:value:0*
T0*A
_output_shapes/
-:+���������������������������c*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:c*
dtype0�
BiasAddBiasAddseparable_conv2d:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������cy
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������c�
NoOpNoOp^BiasAdd/ReadVariableOp ^separable_conv2d/ReadVariableOp"^separable_conv2d/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:+���������������������������J: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2F
!separable_conv2d/ReadVariableOp_1!separable_conv2d/ReadVariableOp_12B
separable_conv2d/ReadVariableOpseparable_conv2d/ReadVariableOp:i e
A
_output_shapes/
-:+���������������������������J
 
_user_specified_nameinputs
�	
�
4__inference_CBlock_1_BatchNorm_layer_call_fn_1690341

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *X
fSRQ
O__inference_CBlock_1_BatchNorm_layer_call_and_return_conditional_losses_1688792�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�

�
B__inference_Conv0_layer_call_and_return_conditional_losses_1690289

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������$,*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������$,g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:���������$,w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������HX: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������HX
 
_user_specified_nameinputs
�

�
D__inference_dense_8_layer_call_and_return_conditional_losses_1689232

inputs0
matmul_readvariableop_resource:c-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:c*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:���������`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������c: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������c
 
_user_specified_nameinputs
��
�0
"__inference__wrapped_model_1688727
input_layerc
Ijojo_n8_r72x88_a0_7_b4_g2_2_strides2_conv0_conv2d_readvariableop_resource:X
Jjojo_n8_r72x88_a0_7_b4_g2_2_strides2_conv0_biasadd_readvariableop_resource:z
`jojo_n8_r72x88_a0_7_b4_g2_2_strides2_cblock_1_sepconv2d_separable_conv2d_readvariableop_resource:|
bjojo_n8_r72x88_a0_7_b4_g2_2_strides2_cblock_1_sepconv2d_separable_conv2d_readvariableop_1_resource:e
Wjojo_n8_r72x88_a0_7_b4_g2_2_strides2_cblock_1_sepconv2d_biasadd_readvariableop_resource:]
Ojojo_n8_r72x88_a0_7_b4_g2_2_strides2_cblock_1_batchnorm_readvariableop_resource:_
Qjojo_n8_r72x88_a0_7_b4_g2_2_strides2_cblock_1_batchnorm_readvariableop_1_resource:n
`jojo_n8_r72x88_a0_7_b4_g2_2_strides2_cblock_1_batchnorm_fusedbatchnormv3_readvariableop_resource:p
bjojo_n8_r72x88_a0_7_b4_g2_2_strides2_cblock_1_batchnorm_fusedbatchnormv3_readvariableop_1_resource:z
`jojo_n8_r72x88_a0_7_b4_g2_2_strides2_cblock_2_sepconv2d_separable_conv2d_readvariableop_resource:|
bjojo_n8_r72x88_a0_7_b4_g2_2_strides2_cblock_2_sepconv2d_separable_conv2d_readvariableop_1_resource:1e
Wjojo_n8_r72x88_a0_7_b4_g2_2_strides2_cblock_2_sepconv2d_biasadd_readvariableop_resource:1]
Ojojo_n8_r72x88_a0_7_b4_g2_2_strides2_cblock_2_batchnorm_readvariableop_resource:1_
Qjojo_n8_r72x88_a0_7_b4_g2_2_strides2_cblock_2_batchnorm_readvariableop_1_resource:1n
`jojo_n8_r72x88_a0_7_b4_g2_2_strides2_cblock_2_batchnorm_fusedbatchnormv3_readvariableop_resource:1p
bjojo_n8_r72x88_a0_7_b4_g2_2_strides2_cblock_2_batchnorm_fusedbatchnormv3_readvariableop_1_resource:1z
`jojo_n8_r72x88_a0_7_b4_g2_2_strides2_cblock_3_sepconv2d_separable_conv2d_readvariableop_resource:1|
bjojo_n8_r72x88_a0_7_b4_g2_2_strides2_cblock_3_sepconv2d_separable_conv2d_readvariableop_1_resource:1Je
Wjojo_n8_r72x88_a0_7_b4_g2_2_strides2_cblock_3_sepconv2d_biasadd_readvariableop_resource:J]
Ojojo_n8_r72x88_a0_7_b4_g2_2_strides2_cblock_3_batchnorm_readvariableop_resource:J_
Qjojo_n8_r72x88_a0_7_b4_g2_2_strides2_cblock_3_batchnorm_readvariableop_1_resource:Jn
`jojo_n8_r72x88_a0_7_b4_g2_2_strides2_cblock_3_batchnorm_fusedbatchnormv3_readvariableop_resource:Jp
bjojo_n8_r72x88_a0_7_b4_g2_2_strides2_cblock_3_batchnorm_fusedbatchnormv3_readvariableop_1_resource:Jz
`jojo_n8_r72x88_a0_7_b4_g2_2_strides2_cblock_4_sepconv2d_separable_conv2d_readvariableop_resource:J|
bjojo_n8_r72x88_a0_7_b4_g2_2_strides2_cblock_4_sepconv2d_separable_conv2d_readvariableop_1_resource:Jce
Wjojo_n8_r72x88_a0_7_b4_g2_2_strides2_cblock_4_sepconv2d_biasadd_readvariableop_resource:c]
Ojojo_n8_r72x88_a0_7_b4_g2_2_strides2_cblock_4_batchnorm_readvariableop_resource:c_
Qjojo_n8_r72x88_a0_7_b4_g2_2_strides2_cblock_4_batchnorm_readvariableop_1_resource:cn
`jojo_n8_r72x88_a0_7_b4_g2_2_strides2_cblock_4_batchnorm_fusedbatchnormv3_readvariableop_resource:cp
bjojo_n8_r72x88_a0_7_b4_g2_2_strides2_cblock_4_batchnorm_fusedbatchnormv3_readvariableop_1_resource:c]
Kjojo_n8_r72x88_a0_7_b4_g2_2_strides2_dense_8_matmul_readvariableop_resource:cZ
Ljojo_n8_r72x88_a0_7_b4_g2_2_strides2_dense_8_biasadd_readvariableop_resource:
identity��Wjojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_1_BatchNorm/FusedBatchNormV3/ReadVariableOp�Yjojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_1_BatchNorm/FusedBatchNormV3/ReadVariableOp_1�Fjojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_1_BatchNorm/ReadVariableOp�Hjojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_1_BatchNorm/ReadVariableOp_1�Njojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_1_SepConv2D/BiasAdd/ReadVariableOp�Wjojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_1_SepConv2D/separable_conv2d/ReadVariableOp�Yjojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_1_SepConv2D/separable_conv2d/ReadVariableOp_1�Wjojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_2_BatchNorm/FusedBatchNormV3/ReadVariableOp�Yjojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_2_BatchNorm/FusedBatchNormV3/ReadVariableOp_1�Fjojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_2_BatchNorm/ReadVariableOp�Hjojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_2_BatchNorm/ReadVariableOp_1�Njojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_2_SepConv2D/BiasAdd/ReadVariableOp�Wjojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_2_SepConv2D/separable_conv2d/ReadVariableOp�Yjojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_2_SepConv2D/separable_conv2d/ReadVariableOp_1�Wjojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_3_BatchNorm/FusedBatchNormV3/ReadVariableOp�Yjojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_3_BatchNorm/FusedBatchNormV3/ReadVariableOp_1�Fjojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_3_BatchNorm/ReadVariableOp�Hjojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_3_BatchNorm/ReadVariableOp_1�Njojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_3_SepConv2D/BiasAdd/ReadVariableOp�Wjojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_3_SepConv2D/separable_conv2d/ReadVariableOp�Yjojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_3_SepConv2D/separable_conv2d/ReadVariableOp_1�Wjojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_4_BatchNorm/FusedBatchNormV3/ReadVariableOp�Yjojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_4_BatchNorm/FusedBatchNormV3/ReadVariableOp_1�Fjojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_4_BatchNorm/ReadVariableOp�Hjojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_4_BatchNorm/ReadVariableOp_1�Njojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_4_SepConv2D/BiasAdd/ReadVariableOp�Wjojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_4_SepConv2D/separable_conv2d/ReadVariableOp�Yjojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_4_SepConv2D/separable_conv2d/ReadVariableOp_1�Ajojo_n8_r72x88_a0.7_b4_g2.2_strides2/Conv0/BiasAdd/ReadVariableOp�@jojo_n8_r72x88_a0.7_b4_g2.2_strides2/Conv0/Conv2D/ReadVariableOp�Cjojo_n8_r72x88_a0.7_b4_g2.2_strides2/dense_8/BiasAdd/ReadVariableOp�Bjojo_n8_r72x88_a0.7_b4_g2.2_strides2/dense_8/MatMul/ReadVariableOp�
@jojo_n8_r72x88_a0.7_b4_g2.2_strides2/Conv0/Conv2D/ReadVariableOpReadVariableOpIjojo_n8_r72x88_a0_7_b4_g2_2_strides2_conv0_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
1jojo_n8_r72x88_a0.7_b4_g2.2_strides2/Conv0/Conv2DConv2Dinput_layerHjojo_n8_r72x88_a0.7_b4_g2.2_strides2/Conv0/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������$,*
paddingSAME*
strides
�
Ajojo_n8_r72x88_a0.7_b4_g2.2_strides2/Conv0/BiasAdd/ReadVariableOpReadVariableOpJjojo_n8_r72x88_a0_7_b4_g2_2_strides2_conv0_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
2jojo_n8_r72x88_a0.7_b4_g2.2_strides2/Conv0/BiasAddBiasAdd:jojo_n8_r72x88_a0.7_b4_g2.2_strides2/Conv0/Conv2D:output:0Ijojo_n8_r72x88_a0.7_b4_g2.2_strides2/Conv0/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������$,�
Wjojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_1_SepConv2D/separable_conv2d/ReadVariableOpReadVariableOp`jojo_n8_r72x88_a0_7_b4_g2_2_strides2_cblock_1_sepconv2d_separable_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
Yjojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_1_SepConv2D/separable_conv2d/ReadVariableOp_1ReadVariableOpbjojo_n8_r72x88_a0_7_b4_g2_2_strides2_cblock_1_sepconv2d_separable_conv2d_readvariableop_1_resource*&
_output_shapes
:*
dtype0�
Njojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_1_SepConv2D/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            �
Vjojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_1_SepConv2D/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      �
Rjojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_1_SepConv2D/separable_conv2d/depthwiseDepthwiseConv2dNative;jojo_n8_r72x88_a0.7_b4_g2.2_strides2/Conv0/BiasAdd:output:0_jojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_1_SepConv2D/separable_conv2d/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
�
Hjojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_1_SepConv2D/separable_conv2dConv2D[jojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_1_SepConv2D/separable_conv2d/depthwise:output:0ajojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_1_SepConv2D/separable_conv2d/ReadVariableOp_1:value:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
�
Njojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_1_SepConv2D/BiasAdd/ReadVariableOpReadVariableOpWjojo_n8_r72x88_a0_7_b4_g2_2_strides2_cblock_1_sepconv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
?jojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_1_SepConv2D/BiasAddBiasAddQjojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_1_SepConv2D/separable_conv2d:output:0Vjojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_1_SepConv2D/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:����������
Fjojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_1_BatchNorm/ReadVariableOpReadVariableOpOjojo_n8_r72x88_a0_7_b4_g2_2_strides2_cblock_1_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0�
Hjojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_1_BatchNorm/ReadVariableOp_1ReadVariableOpQjojo_n8_r72x88_a0_7_b4_g2_2_strides2_cblock_1_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0�
Wjojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_1_BatchNorm/FusedBatchNormV3/ReadVariableOpReadVariableOp`jojo_n8_r72x88_a0_7_b4_g2_2_strides2_cblock_1_batchnorm_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0�
Yjojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_1_BatchNorm/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpbjojo_n8_r72x88_a0_7_b4_g2_2_strides2_cblock_1_batchnorm_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0�
Hjojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_1_BatchNorm/FusedBatchNormV3FusedBatchNormV3Hjojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_1_SepConv2D/BiasAdd:output:0Njojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_1_BatchNorm/ReadVariableOp:value:0Pjojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_1_BatchNorm/ReadVariableOp_1:value:0_jojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_1_BatchNorm/FusedBatchNormV3/ReadVariableOp:value:0ajojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_1_BatchNorm/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������:::::*
epsilon%o�:*
is_training( �
8jojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_1_ReLu/Relu6Relu6Ljojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_1_BatchNorm/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:����������
Wjojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_2_SepConv2D/separable_conv2d/ReadVariableOpReadVariableOp`jojo_n8_r72x88_a0_7_b4_g2_2_strides2_cblock_2_sepconv2d_separable_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
Yjojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_2_SepConv2D/separable_conv2d/ReadVariableOp_1ReadVariableOpbjojo_n8_r72x88_a0_7_b4_g2_2_strides2_cblock_2_sepconv2d_separable_conv2d_readvariableop_1_resource*&
_output_shapes
:1*
dtype0�
Njojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_2_SepConv2D/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            �
Vjojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_2_SepConv2D/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      �
Rjojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_2_SepConv2D/separable_conv2d/depthwiseDepthwiseConv2dNativeFjojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_1_ReLu/Relu6:activations:0_jojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_2_SepConv2D/separable_conv2d/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������	*
paddingSAME*
strides
�
Hjojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_2_SepConv2D/separable_conv2dConv2D[jojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_2_SepConv2D/separable_conv2d/depthwise:output:0ajojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_2_SepConv2D/separable_conv2d/ReadVariableOp_1:value:0*
T0*/
_output_shapes
:���������	1*
paddingVALID*
strides
�
Njojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_2_SepConv2D/BiasAdd/ReadVariableOpReadVariableOpWjojo_n8_r72x88_a0_7_b4_g2_2_strides2_cblock_2_sepconv2d_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype0�
?jojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_2_SepConv2D/BiasAddBiasAddQjojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_2_SepConv2D/separable_conv2d:output:0Vjojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_2_SepConv2D/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������	1�
Fjojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_2_BatchNorm/ReadVariableOpReadVariableOpOjojo_n8_r72x88_a0_7_b4_g2_2_strides2_cblock_2_batchnorm_readvariableop_resource*
_output_shapes
:1*
dtype0�
Hjojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_2_BatchNorm/ReadVariableOp_1ReadVariableOpQjojo_n8_r72x88_a0_7_b4_g2_2_strides2_cblock_2_batchnorm_readvariableop_1_resource*
_output_shapes
:1*
dtype0�
Wjojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_2_BatchNorm/FusedBatchNormV3/ReadVariableOpReadVariableOp`jojo_n8_r72x88_a0_7_b4_g2_2_strides2_cblock_2_batchnorm_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:1*
dtype0�
Yjojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_2_BatchNorm/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpbjojo_n8_r72x88_a0_7_b4_g2_2_strides2_cblock_2_batchnorm_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:1*
dtype0�
Hjojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_2_BatchNorm/FusedBatchNormV3FusedBatchNormV3Hjojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_2_SepConv2D/BiasAdd:output:0Njojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_2_BatchNorm/ReadVariableOp:value:0Pjojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_2_BatchNorm/ReadVariableOp_1:value:0_jojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_2_BatchNorm/FusedBatchNormV3/ReadVariableOp:value:0ajojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_2_BatchNorm/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������	1:1:1:1:1:*
epsilon%o�:*
is_training( �
8jojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_2_ReLu/Relu6Relu6Ljojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_2_BatchNorm/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:���������	1�
Wjojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_3_SepConv2D/separable_conv2d/ReadVariableOpReadVariableOp`jojo_n8_r72x88_a0_7_b4_g2_2_strides2_cblock_3_sepconv2d_separable_conv2d_readvariableop_resource*&
_output_shapes
:1*
dtype0�
Yjojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_3_SepConv2D/separable_conv2d/ReadVariableOp_1ReadVariableOpbjojo_n8_r72x88_a0_7_b4_g2_2_strides2_cblock_3_sepconv2d_separable_conv2d_readvariableop_1_resource*&
_output_shapes
:1J*
dtype0�
Njojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_3_SepConv2D/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      1      �
Vjojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_3_SepConv2D/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      �
Rjojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_3_SepConv2D/separable_conv2d/depthwiseDepthwiseConv2dNativeFjojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_2_ReLu/Relu6:activations:0_jojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_3_SepConv2D/separable_conv2d/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������1*
paddingSAME*
strides
�
Hjojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_3_SepConv2D/separable_conv2dConv2D[jojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_3_SepConv2D/separable_conv2d/depthwise:output:0ajojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_3_SepConv2D/separable_conv2d/ReadVariableOp_1:value:0*
T0*/
_output_shapes
:���������J*
paddingVALID*
strides
�
Njojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_3_SepConv2D/BiasAdd/ReadVariableOpReadVariableOpWjojo_n8_r72x88_a0_7_b4_g2_2_strides2_cblock_3_sepconv2d_biasadd_readvariableop_resource*
_output_shapes
:J*
dtype0�
?jojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_3_SepConv2D/BiasAddBiasAddQjojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_3_SepConv2D/separable_conv2d:output:0Vjojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_3_SepConv2D/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������J�
Fjojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_3_BatchNorm/ReadVariableOpReadVariableOpOjojo_n8_r72x88_a0_7_b4_g2_2_strides2_cblock_3_batchnorm_readvariableop_resource*
_output_shapes
:J*
dtype0�
Hjojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_3_BatchNorm/ReadVariableOp_1ReadVariableOpQjojo_n8_r72x88_a0_7_b4_g2_2_strides2_cblock_3_batchnorm_readvariableop_1_resource*
_output_shapes
:J*
dtype0�
Wjojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_3_BatchNorm/FusedBatchNormV3/ReadVariableOpReadVariableOp`jojo_n8_r72x88_a0_7_b4_g2_2_strides2_cblock_3_batchnorm_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:J*
dtype0�
Yjojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_3_BatchNorm/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpbjojo_n8_r72x88_a0_7_b4_g2_2_strides2_cblock_3_batchnorm_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:J*
dtype0�
Hjojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_3_BatchNorm/FusedBatchNormV3FusedBatchNormV3Hjojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_3_SepConv2D/BiasAdd:output:0Njojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_3_BatchNorm/ReadVariableOp:value:0Pjojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_3_BatchNorm/ReadVariableOp_1:value:0_jojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_3_BatchNorm/FusedBatchNormV3/ReadVariableOp:value:0ajojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_3_BatchNorm/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������J:J:J:J:J:*
epsilon%o�:*
is_training( �
8jojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_3_ReLu/Relu6Relu6Ljojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_3_BatchNorm/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:���������J�
Wjojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_4_SepConv2D/separable_conv2d/ReadVariableOpReadVariableOp`jojo_n8_r72x88_a0_7_b4_g2_2_strides2_cblock_4_sepconv2d_separable_conv2d_readvariableop_resource*&
_output_shapes
:J*
dtype0�
Yjojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_4_SepConv2D/separable_conv2d/ReadVariableOp_1ReadVariableOpbjojo_n8_r72x88_a0_7_b4_g2_2_strides2_cblock_4_sepconv2d_separable_conv2d_readvariableop_1_resource*&
_output_shapes
:Jc*
dtype0�
Njojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_4_SepConv2D/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      J      �
Vjojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_4_SepConv2D/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      �
Rjojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_4_SepConv2D/separable_conv2d/depthwiseDepthwiseConv2dNativeFjojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_3_ReLu/Relu6:activations:0_jojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_4_SepConv2D/separable_conv2d/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������J*
paddingSAME*
strides
�
Hjojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_4_SepConv2D/separable_conv2dConv2D[jojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_4_SepConv2D/separable_conv2d/depthwise:output:0ajojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_4_SepConv2D/separable_conv2d/ReadVariableOp_1:value:0*
T0*/
_output_shapes
:���������c*
paddingVALID*
strides
�
Njojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_4_SepConv2D/BiasAdd/ReadVariableOpReadVariableOpWjojo_n8_r72x88_a0_7_b4_g2_2_strides2_cblock_4_sepconv2d_biasadd_readvariableop_resource*
_output_shapes
:c*
dtype0�
?jojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_4_SepConv2D/BiasAddBiasAddQjojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_4_SepConv2D/separable_conv2d:output:0Vjojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_4_SepConv2D/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������c�
Fjojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_4_BatchNorm/ReadVariableOpReadVariableOpOjojo_n8_r72x88_a0_7_b4_g2_2_strides2_cblock_4_batchnorm_readvariableop_resource*
_output_shapes
:c*
dtype0�
Hjojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_4_BatchNorm/ReadVariableOp_1ReadVariableOpQjojo_n8_r72x88_a0_7_b4_g2_2_strides2_cblock_4_batchnorm_readvariableop_1_resource*
_output_shapes
:c*
dtype0�
Wjojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_4_BatchNorm/FusedBatchNormV3/ReadVariableOpReadVariableOp`jojo_n8_r72x88_a0_7_b4_g2_2_strides2_cblock_4_batchnorm_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:c*
dtype0�
Yjojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_4_BatchNorm/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpbjojo_n8_r72x88_a0_7_b4_g2_2_strides2_cblock_4_batchnorm_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:c*
dtype0�
Hjojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_4_BatchNorm/FusedBatchNormV3FusedBatchNormV3Hjojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_4_SepConv2D/BiasAdd:output:0Njojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_4_BatchNorm/ReadVariableOp:value:0Pjojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_4_BatchNorm/ReadVariableOp_1:value:0_jojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_4_BatchNorm/FusedBatchNormV3/ReadVariableOp:value:0ajojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_4_BatchNorm/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������c:c:c:c:c:*
epsilon%o�:*
is_training( �
8jojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_4_ReLu/Relu6Relu6Ljojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_4_BatchNorm/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:���������c�
Vjojo_n8_r72x88_a0.7_b4_g2.2_strides2/global_average_pooling2d_8/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      �
Djojo_n8_r72x88_a0.7_b4_g2.2_strides2/global_average_pooling2d_8/MeanMeanFjojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_4_ReLu/Relu6:activations:0_jojo_n8_r72x88_a0.7_b4_g2.2_strides2/global_average_pooling2d_8/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:���������c�
Bjojo_n8_r72x88_a0.7_b4_g2.2_strides2/dense_8/MatMul/ReadVariableOpReadVariableOpKjojo_n8_r72x88_a0_7_b4_g2_2_strides2_dense_8_matmul_readvariableop_resource*
_output_shapes

:c*
dtype0�
3jojo_n8_r72x88_a0.7_b4_g2.2_strides2/dense_8/MatMulMatMulMjojo_n8_r72x88_a0.7_b4_g2.2_strides2/global_average_pooling2d_8/Mean:output:0Jjojo_n8_r72x88_a0.7_b4_g2.2_strides2/dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
Cjojo_n8_r72x88_a0.7_b4_g2.2_strides2/dense_8/BiasAdd/ReadVariableOpReadVariableOpLjojo_n8_r72x88_a0_7_b4_g2_2_strides2_dense_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
4jojo_n8_r72x88_a0.7_b4_g2.2_strides2/dense_8/BiasAddBiasAdd=jojo_n8_r72x88_a0.7_b4_g2.2_strides2/dense_8/MatMul:product:0Kjojo_n8_r72x88_a0.7_b4_g2.2_strides2/dense_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
4jojo_n8_r72x88_a0.7_b4_g2.2_strides2/dense_8/SoftmaxSoftmax=jojo_n8_r72x88_a0.7_b4_g2.2_strides2/dense_8/BiasAdd:output:0*
T0*'
_output_shapes
:����������
IdentityIdentity>jojo_n8_r72x88_a0.7_b4_g2.2_strides2/dense_8/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOpX^jojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_1_BatchNorm/FusedBatchNormV3/ReadVariableOpZ^jojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_1_BatchNorm/FusedBatchNormV3/ReadVariableOp_1G^jojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_1_BatchNorm/ReadVariableOpI^jojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_1_BatchNorm/ReadVariableOp_1O^jojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_1_SepConv2D/BiasAdd/ReadVariableOpX^jojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_1_SepConv2D/separable_conv2d/ReadVariableOpZ^jojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_1_SepConv2D/separable_conv2d/ReadVariableOp_1X^jojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_2_BatchNorm/FusedBatchNormV3/ReadVariableOpZ^jojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_2_BatchNorm/FusedBatchNormV3/ReadVariableOp_1G^jojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_2_BatchNorm/ReadVariableOpI^jojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_2_BatchNorm/ReadVariableOp_1O^jojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_2_SepConv2D/BiasAdd/ReadVariableOpX^jojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_2_SepConv2D/separable_conv2d/ReadVariableOpZ^jojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_2_SepConv2D/separable_conv2d/ReadVariableOp_1X^jojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_3_BatchNorm/FusedBatchNormV3/ReadVariableOpZ^jojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_3_BatchNorm/FusedBatchNormV3/ReadVariableOp_1G^jojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_3_BatchNorm/ReadVariableOpI^jojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_3_BatchNorm/ReadVariableOp_1O^jojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_3_SepConv2D/BiasAdd/ReadVariableOpX^jojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_3_SepConv2D/separable_conv2d/ReadVariableOpZ^jojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_3_SepConv2D/separable_conv2d/ReadVariableOp_1X^jojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_4_BatchNorm/FusedBatchNormV3/ReadVariableOpZ^jojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_4_BatchNorm/FusedBatchNormV3/ReadVariableOp_1G^jojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_4_BatchNorm/ReadVariableOpI^jojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_4_BatchNorm/ReadVariableOp_1O^jojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_4_SepConv2D/BiasAdd/ReadVariableOpX^jojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_4_SepConv2D/separable_conv2d/ReadVariableOpZ^jojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_4_SepConv2D/separable_conv2d/ReadVariableOp_1B^jojo_n8_r72x88_a0.7_b4_g2.2_strides2/Conv0/BiasAdd/ReadVariableOpA^jojo_n8_r72x88_a0.7_b4_g2.2_strides2/Conv0/Conv2D/ReadVariableOpD^jojo_n8_r72x88_a0.7_b4_g2.2_strides2/dense_8/BiasAdd/ReadVariableOpC^jojo_n8_r72x88_a0.7_b4_g2.2_strides2/dense_8/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*n
_input_shapes]
[:���������HX: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2�
Yjojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_1_BatchNorm/FusedBatchNormV3/ReadVariableOp_1Yjojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_1_BatchNorm/FusedBatchNormV3/ReadVariableOp_12�
Wjojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_1_BatchNorm/FusedBatchNormV3/ReadVariableOpWjojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_1_BatchNorm/FusedBatchNormV3/ReadVariableOp2�
Hjojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_1_BatchNorm/ReadVariableOp_1Hjojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_1_BatchNorm/ReadVariableOp_12�
Fjojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_1_BatchNorm/ReadVariableOpFjojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_1_BatchNorm/ReadVariableOp2�
Njojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_1_SepConv2D/BiasAdd/ReadVariableOpNjojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_1_SepConv2D/BiasAdd/ReadVariableOp2�
Yjojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_1_SepConv2D/separable_conv2d/ReadVariableOp_1Yjojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_1_SepConv2D/separable_conv2d/ReadVariableOp_12�
Wjojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_1_SepConv2D/separable_conv2d/ReadVariableOpWjojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_1_SepConv2D/separable_conv2d/ReadVariableOp2�
Yjojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_2_BatchNorm/FusedBatchNormV3/ReadVariableOp_1Yjojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_2_BatchNorm/FusedBatchNormV3/ReadVariableOp_12�
Wjojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_2_BatchNorm/FusedBatchNormV3/ReadVariableOpWjojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_2_BatchNorm/FusedBatchNormV3/ReadVariableOp2�
Hjojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_2_BatchNorm/ReadVariableOp_1Hjojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_2_BatchNorm/ReadVariableOp_12�
Fjojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_2_BatchNorm/ReadVariableOpFjojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_2_BatchNorm/ReadVariableOp2�
Njojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_2_SepConv2D/BiasAdd/ReadVariableOpNjojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_2_SepConv2D/BiasAdd/ReadVariableOp2�
Yjojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_2_SepConv2D/separable_conv2d/ReadVariableOp_1Yjojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_2_SepConv2D/separable_conv2d/ReadVariableOp_12�
Wjojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_2_SepConv2D/separable_conv2d/ReadVariableOpWjojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_2_SepConv2D/separable_conv2d/ReadVariableOp2�
Yjojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_3_BatchNorm/FusedBatchNormV3/ReadVariableOp_1Yjojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_3_BatchNorm/FusedBatchNormV3/ReadVariableOp_12�
Wjojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_3_BatchNorm/FusedBatchNormV3/ReadVariableOpWjojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_3_BatchNorm/FusedBatchNormV3/ReadVariableOp2�
Hjojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_3_BatchNorm/ReadVariableOp_1Hjojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_3_BatchNorm/ReadVariableOp_12�
Fjojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_3_BatchNorm/ReadVariableOpFjojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_3_BatchNorm/ReadVariableOp2�
Njojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_3_SepConv2D/BiasAdd/ReadVariableOpNjojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_3_SepConv2D/BiasAdd/ReadVariableOp2�
Yjojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_3_SepConv2D/separable_conv2d/ReadVariableOp_1Yjojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_3_SepConv2D/separable_conv2d/ReadVariableOp_12�
Wjojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_3_SepConv2D/separable_conv2d/ReadVariableOpWjojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_3_SepConv2D/separable_conv2d/ReadVariableOp2�
Yjojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_4_BatchNorm/FusedBatchNormV3/ReadVariableOp_1Yjojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_4_BatchNorm/FusedBatchNormV3/ReadVariableOp_12�
Wjojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_4_BatchNorm/FusedBatchNormV3/ReadVariableOpWjojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_4_BatchNorm/FusedBatchNormV3/ReadVariableOp2�
Hjojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_4_BatchNorm/ReadVariableOp_1Hjojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_4_BatchNorm/ReadVariableOp_12�
Fjojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_4_BatchNorm/ReadVariableOpFjojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_4_BatchNorm/ReadVariableOp2�
Njojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_4_SepConv2D/BiasAdd/ReadVariableOpNjojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_4_SepConv2D/BiasAdd/ReadVariableOp2�
Yjojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_4_SepConv2D/separable_conv2d/ReadVariableOp_1Yjojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_4_SepConv2D/separable_conv2d/ReadVariableOp_12�
Wjojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_4_SepConv2D/separable_conv2d/ReadVariableOpWjojo_n8_r72x88_a0.7_b4_g2.2_strides2/CBlock_4_SepConv2D/separable_conv2d/ReadVariableOp2�
Ajojo_n8_r72x88_a0.7_b4_g2.2_strides2/Conv0/BiasAdd/ReadVariableOpAjojo_n8_r72x88_a0.7_b4_g2.2_strides2/Conv0/BiasAdd/ReadVariableOp2�
@jojo_n8_r72x88_a0.7_b4_g2.2_strides2/Conv0/Conv2D/ReadVariableOp@jojo_n8_r72x88_a0.7_b4_g2.2_strides2/Conv0/Conv2D/ReadVariableOp2�
Cjojo_n8_r72x88_a0.7_b4_g2.2_strides2/dense_8/BiasAdd/ReadVariableOpCjojo_n8_r72x88_a0.7_b4_g2.2_strides2/dense_8/BiasAdd/ReadVariableOp2�
Bjojo_n8_r72x88_a0.7_b4_g2.2_strides2/dense_8/MatMul/ReadVariableOpBjojo_n8_r72x88_a0.7_b4_g2.2_strides2/dense_8/MatMul/ReadVariableOp:\ X
/
_output_shapes
:���������HX
%
_user_specified_nameinput_layer
�
�
O__inference_CBlock_2_BatchNorm_layer_call_and_return_conditional_losses_1688884

inputs%
readvariableop_resource:1'
readvariableop_1_resource:16
(fusedbatchnormv3_readvariableop_resource:18
*fusedbatchnormv3_readvariableop_1_resource:1
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:1*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:1*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:1*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:1*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������1:1:1:1:1:*
epsilon%o�:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������1�
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������1: : : : 2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:i e
A
_output_shapes/
-:+���������������������������1
 
_user_specified_nameinputs
�	
�
4__inference_CBlock_2_BatchNorm_layer_call_fn_1690426

inputs
unknown:1
	unknown_0:1
	unknown_1:1
	unknown_2:1
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������1*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *X
fSRQ
O__inference_CBlock_2_BatchNorm_layer_call_and_return_conditional_losses_1688866�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������1`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������1: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������1
 
_user_specified_nameinputs
�
K
/__inference_CBlock_1_ReLu_layer_call_fn_1690382

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_CBlock_1_ReLu_layer_call_and_return_conditional_losses_1689149h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
%__inference_signature_wrapper_1689886
input_layer!
unknown:
	unknown_0:#
	unknown_1:#
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:#
	unknown_8:#
	unknown_9:1

unknown_10:1

unknown_11:1

unknown_12:1

unknown_13:1

unknown_14:1$

unknown_15:1$

unknown_16:1J

unknown_17:J

unknown_18:J

unknown_19:J

unknown_20:J

unknown_21:J$

unknown_22:J$

unknown_23:Jc

unknown_24:c

unknown_25:c

unknown_26:c

unknown_27:c

unknown_28:c

unknown_29:c

unknown_30:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_layerunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30*,
Tin%
#2!*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*B
_read_only_resource_inputs$
" 	
 *0
config_proto 

CPU

GPU2*0J 8� *+
f&R$
"__inference__wrapped_model_1688727o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*n
_input_shapes]
[:���������HX: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
/
_output_shapes
:���������HX
%
_user_specified_nameinput_layer
�
�
O__inference_CBlock_2_SepConv2D_layer_call_and_return_conditional_losses_1690413

inputsB
(separable_conv2d_readvariableop_resource:D
*separable_conv2d_readvariableop_1_resource:1-
biasadd_readvariableop_resource:1
identity��BiasAdd/ReadVariableOp�separable_conv2d/ReadVariableOp�!separable_conv2d/ReadVariableOp_1�
separable_conv2d/ReadVariableOpReadVariableOp(separable_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
!separable_conv2d/ReadVariableOp_1ReadVariableOp*separable_conv2d_readvariableop_1_resource*&
_output_shapes
:1*
dtype0o
separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            o
separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      �
separable_conv2d/depthwiseDepthwiseConv2dNativeinputs'separable_conv2d/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������*
paddingSAME*
strides
�
separable_conv2dConv2D#separable_conv2d/depthwise:output:0)separable_conv2d/ReadVariableOp_1:value:0*
T0*A
_output_shapes/
-:+���������������������������1*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:1*
dtype0�
BiasAddBiasAddseparable_conv2d:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������1y
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������1�
NoOpNoOp^BiasAdd/ReadVariableOp ^separable_conv2d/ReadVariableOp"^separable_conv2d/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:+���������������������������: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2F
!separable_conv2d/ReadVariableOp_1!separable_conv2d/ReadVariableOp_12B
separable_conv2d/ReadVariableOpseparable_conv2d/ReadVariableOp:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
�
O__inference_CBlock_2_BatchNorm_layer_call_and_return_conditional_losses_1690457

inputs%
readvariableop_resource:1'
readvariableop_1_resource:16
(fusedbatchnormv3_readvariableop_resource:18
*fusedbatchnormv3_readvariableop_1_resource:1
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:1*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:1*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:1*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:1*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������1:1:1:1:1:*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������1�
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������1: : : : 2$
AssignNewValue_1AssignNewValue_12 
AssignNewValueAssignNewValue2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:i e
A
_output_shapes/
-:+���������������������������1
 
_user_specified_nameinputs
�
�
O__inference_CBlock_3_SepConv2D_layer_call_and_return_conditional_losses_1688927

inputsB
(separable_conv2d_readvariableop_resource:1D
*separable_conv2d_readvariableop_1_resource:1J-
biasadd_readvariableop_resource:J
identity��BiasAdd/ReadVariableOp�separable_conv2d/ReadVariableOp�!separable_conv2d/ReadVariableOp_1�
separable_conv2d/ReadVariableOpReadVariableOp(separable_conv2d_readvariableop_resource*&
_output_shapes
:1*
dtype0�
!separable_conv2d/ReadVariableOp_1ReadVariableOp*separable_conv2d_readvariableop_1_resource*&
_output_shapes
:1J*
dtype0o
separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      1      o
separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      �
separable_conv2d/depthwiseDepthwiseConv2dNativeinputs'separable_conv2d/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������1*
paddingSAME*
strides
�
separable_conv2dConv2D#separable_conv2d/depthwise:output:0)separable_conv2d/ReadVariableOp_1:value:0*
T0*A
_output_shapes/
-:+���������������������������J*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:J*
dtype0�
BiasAddBiasAddseparable_conv2d:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������Jy
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������J�
NoOpNoOp^BiasAdd/ReadVariableOp ^separable_conv2d/ReadVariableOp"^separable_conv2d/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:+���������������������������1: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2F
!separable_conv2d/ReadVariableOp_1!separable_conv2d/ReadVariableOp_12B
separable_conv2d/ReadVariableOpseparable_conv2d/ReadVariableOp:i e
A
_output_shapes/
-:+���������������������������1
 
_user_specified_nameinputs
�	
�
4__inference_CBlock_3_SepConv2D_layer_call_fn_1690496

inputs!
unknown:1#
	unknown_0:1J
	unknown_1:J
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������J*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *X
fSRQ
O__inference_CBlock_3_SepConv2D_layer_call_and_return_conditional_losses_1688927�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������J`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:+���������������������������1: : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������1
 
_user_specified_nameinputs
�R
�
a__inference_jojo_n8_r72x88_a0.7_b4_g2.2_strides2_layer_call_and_return_conditional_losses_1689322
input_layer'
conv0_1689242:
conv0_1689244:4
cblock_1_sepconv2d_1689247:4
cblock_1_sepconv2d_1689249:(
cblock_1_sepconv2d_1689251:(
cblock_1_batchnorm_1689254:(
cblock_1_batchnorm_1689256:(
cblock_1_batchnorm_1689258:(
cblock_1_batchnorm_1689260:4
cblock_2_sepconv2d_1689264:4
cblock_2_sepconv2d_1689266:1(
cblock_2_sepconv2d_1689268:1(
cblock_2_batchnorm_1689271:1(
cblock_2_batchnorm_1689273:1(
cblock_2_batchnorm_1689275:1(
cblock_2_batchnorm_1689277:14
cblock_3_sepconv2d_1689281:14
cblock_3_sepconv2d_1689283:1J(
cblock_3_sepconv2d_1689285:J(
cblock_3_batchnorm_1689288:J(
cblock_3_batchnorm_1689290:J(
cblock_3_batchnorm_1689292:J(
cblock_3_batchnorm_1689294:J4
cblock_4_sepconv2d_1689298:J4
cblock_4_sepconv2d_1689300:Jc(
cblock_4_sepconv2d_1689302:c(
cblock_4_batchnorm_1689305:c(
cblock_4_batchnorm_1689307:c(
cblock_4_batchnorm_1689309:c(
cblock_4_batchnorm_1689311:c!
dense_8_1689316:c
dense_8_1689318:
identity��*CBlock_1_BatchNorm/StatefulPartitionedCall�*CBlock_1_SepConv2D/StatefulPartitionedCall�*CBlock_2_BatchNorm/StatefulPartitionedCall�*CBlock_2_SepConv2D/StatefulPartitionedCall�*CBlock_3_BatchNorm/StatefulPartitionedCall�*CBlock_3_SepConv2D/StatefulPartitionedCall�*CBlock_4_BatchNorm/StatefulPartitionedCall�*CBlock_4_SepConv2D/StatefulPartitionedCall�Conv0/StatefulPartitionedCall�dense_8/StatefulPartitionedCall�
Conv0/StatefulPartitionedCallStatefulPartitionedCallinput_layerconv0_1689242conv0_1689244*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������$,*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_Conv0_layer_call_and_return_conditional_losses_1689122�
*CBlock_1_SepConv2D/StatefulPartitionedCallStatefulPartitionedCall&Conv0/StatefulPartitionedCall:output:0cblock_1_sepconv2d_1689247cblock_1_sepconv2d_1689249cblock_1_sepconv2d_1689251*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *X
fSRQ
O__inference_CBlock_1_SepConv2D_layer_call_and_return_conditional_losses_1688743�
*CBlock_1_BatchNorm/StatefulPartitionedCallStatefulPartitionedCall3CBlock_1_SepConv2D/StatefulPartitionedCall:output:0cblock_1_batchnorm_1689254cblock_1_batchnorm_1689256cblock_1_batchnorm_1689258cblock_1_batchnorm_1689260*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *X
fSRQ
O__inference_CBlock_1_BatchNorm_layer_call_and_return_conditional_losses_1688792�
CBlock_1_ReLu/PartitionedCallPartitionedCall3CBlock_1_BatchNorm/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_CBlock_1_ReLu_layer_call_and_return_conditional_losses_1689149�
*CBlock_2_SepConv2D/StatefulPartitionedCallStatefulPartitionedCall&CBlock_1_ReLu/PartitionedCall:output:0cblock_2_sepconv2d_1689264cblock_2_sepconv2d_1689266cblock_2_sepconv2d_1689268*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������	1*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *X
fSRQ
O__inference_CBlock_2_SepConv2D_layer_call_and_return_conditional_losses_1688835�
*CBlock_2_BatchNorm/StatefulPartitionedCallStatefulPartitionedCall3CBlock_2_SepConv2D/StatefulPartitionedCall:output:0cblock_2_batchnorm_1689271cblock_2_batchnorm_1689273cblock_2_batchnorm_1689275cblock_2_batchnorm_1689277*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������	1*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *X
fSRQ
O__inference_CBlock_2_BatchNorm_layer_call_and_return_conditional_losses_1688884�
CBlock_2_ReLu/PartitionedCallPartitionedCall3CBlock_2_BatchNorm/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������	1* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_CBlock_2_ReLu_layer_call_and_return_conditional_losses_1689172�
*CBlock_3_SepConv2D/StatefulPartitionedCallStatefulPartitionedCall&CBlock_2_ReLu/PartitionedCall:output:0cblock_3_sepconv2d_1689281cblock_3_sepconv2d_1689283cblock_3_sepconv2d_1689285*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������J*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *X
fSRQ
O__inference_CBlock_3_SepConv2D_layer_call_and_return_conditional_losses_1688927�
*CBlock_3_BatchNorm/StatefulPartitionedCallStatefulPartitionedCall3CBlock_3_SepConv2D/StatefulPartitionedCall:output:0cblock_3_batchnorm_1689288cblock_3_batchnorm_1689290cblock_3_batchnorm_1689292cblock_3_batchnorm_1689294*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������J*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *X
fSRQ
O__inference_CBlock_3_BatchNorm_layer_call_and_return_conditional_losses_1688976�
CBlock_3_ReLu/PartitionedCallPartitionedCall3CBlock_3_BatchNorm/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������J* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_CBlock_3_ReLu_layer_call_and_return_conditional_losses_1689195�
*CBlock_4_SepConv2D/StatefulPartitionedCallStatefulPartitionedCall&CBlock_3_ReLu/PartitionedCall:output:0cblock_4_sepconv2d_1689298cblock_4_sepconv2d_1689300cblock_4_sepconv2d_1689302*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������c*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *X
fSRQ
O__inference_CBlock_4_SepConv2D_layer_call_and_return_conditional_losses_1689019�
*CBlock_4_BatchNorm/StatefulPartitionedCallStatefulPartitionedCall3CBlock_4_SepConv2D/StatefulPartitionedCall:output:0cblock_4_batchnorm_1689305cblock_4_batchnorm_1689307cblock_4_batchnorm_1689309cblock_4_batchnorm_1689311*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������c*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *X
fSRQ
O__inference_CBlock_4_BatchNorm_layer_call_and_return_conditional_losses_1689068�
CBlock_4_ReLu/PartitionedCallPartitionedCall3CBlock_4_BatchNorm/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������c* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_CBlock_4_ReLu_layer_call_and_return_conditional_losses_1689218�
*global_average_pooling2d_8/PartitionedCallPartitionedCall&CBlock_4_ReLu/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������c* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *`
f[RY
W__inference_global_average_pooling2d_8_layer_call_and_return_conditional_losses_1689102�
dense_8/StatefulPartitionedCallStatefulPartitionedCall3global_average_pooling2d_8/PartitionedCall:output:0dense_8_1689316dense_8_1689318*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_dense_8_layer_call_and_return_conditional_losses_1689232w
IdentityIdentity(dense_8/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp+^CBlock_1_BatchNorm/StatefulPartitionedCall+^CBlock_1_SepConv2D/StatefulPartitionedCall+^CBlock_2_BatchNorm/StatefulPartitionedCall+^CBlock_2_SepConv2D/StatefulPartitionedCall+^CBlock_3_BatchNorm/StatefulPartitionedCall+^CBlock_3_SepConv2D/StatefulPartitionedCall+^CBlock_4_BatchNorm/StatefulPartitionedCall+^CBlock_4_SepConv2D/StatefulPartitionedCall^Conv0/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*n
_input_shapes]
[:���������HX: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2X
*CBlock_1_BatchNorm/StatefulPartitionedCall*CBlock_1_BatchNorm/StatefulPartitionedCall2X
*CBlock_1_SepConv2D/StatefulPartitionedCall*CBlock_1_SepConv2D/StatefulPartitionedCall2X
*CBlock_2_BatchNorm/StatefulPartitionedCall*CBlock_2_BatchNorm/StatefulPartitionedCall2X
*CBlock_2_SepConv2D/StatefulPartitionedCall*CBlock_2_SepConv2D/StatefulPartitionedCall2X
*CBlock_3_BatchNorm/StatefulPartitionedCall*CBlock_3_BatchNorm/StatefulPartitionedCall2X
*CBlock_3_SepConv2D/StatefulPartitionedCall*CBlock_3_SepConv2D/StatefulPartitionedCall2X
*CBlock_4_BatchNorm/StatefulPartitionedCall*CBlock_4_BatchNorm/StatefulPartitionedCall2X
*CBlock_4_SepConv2D/StatefulPartitionedCall*CBlock_4_SepConv2D/StatefulPartitionedCall2>
Conv0/StatefulPartitionedCallConv0/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall:\ X
/
_output_shapes
:���������HX
%
_user_specified_nameinput_layer
�	
�
4__inference_CBlock_4_SepConv2D_layer_call_fn_1690594

inputs!
unknown:J#
	unknown_0:Jc
	unknown_1:c
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������c*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *X
fSRQ
O__inference_CBlock_4_SepConv2D_layer_call_and_return_conditional_losses_1689019�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������c`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:+���������������������������J: : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������J
 
_user_specified_nameinputs
�
f
J__inference_CBlock_1_ReLu_layer_call_and_return_conditional_losses_1690387

inputs
identityP
Relu6Relu6inputs*
T0*/
_output_shapes
:���������c
IdentityIdentityRelu6:activations:0*
T0*/
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
O__inference_CBlock_4_BatchNorm_layer_call_and_return_conditional_losses_1690653

inputs%
readvariableop_resource:c'
readvariableop_1_resource:c6
(fusedbatchnormv3_readvariableop_resource:c8
*fusedbatchnormv3_readvariableop_1_resource:c
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:c*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:c*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:c*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:c*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������c:c:c:c:c:*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������c�
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������c: : : : 2$
AssignNewValue_1AssignNewValue_12 
AssignNewValueAssignNewValue2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:i e
A
_output_shapes/
-:+���������������������������c
 
_user_specified_nameinputs
�
�
O__inference_CBlock_2_SepConv2D_layer_call_and_return_conditional_losses_1688835

inputsB
(separable_conv2d_readvariableop_resource:D
*separable_conv2d_readvariableop_1_resource:1-
biasadd_readvariableop_resource:1
identity��BiasAdd/ReadVariableOp�separable_conv2d/ReadVariableOp�!separable_conv2d/ReadVariableOp_1�
separable_conv2d/ReadVariableOpReadVariableOp(separable_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
!separable_conv2d/ReadVariableOp_1ReadVariableOp*separable_conv2d_readvariableop_1_resource*&
_output_shapes
:1*
dtype0o
separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            o
separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      �
separable_conv2d/depthwiseDepthwiseConv2dNativeinputs'separable_conv2d/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������*
paddingSAME*
strides
�
separable_conv2dConv2D#separable_conv2d/depthwise:output:0)separable_conv2d/ReadVariableOp_1:value:0*
T0*A
_output_shapes/
-:+���������������������������1*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:1*
dtype0�
BiasAddBiasAddseparable_conv2d:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������1y
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������1�
NoOpNoOp^BiasAdd/ReadVariableOp ^separable_conv2d/ReadVariableOp"^separable_conv2d/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:+���������������������������: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2F
!separable_conv2d/ReadVariableOp_1!separable_conv2d/ReadVariableOp_12B
separable_conv2d/ReadVariableOpseparable_conv2d/ReadVariableOp:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
f
J__inference_CBlock_4_ReLu_layer_call_and_return_conditional_losses_1690681

inputs
identityP
Relu6Relu6inputs*
T0*/
_output_shapes
:���������cc
IdentityIdentityRelu6:activations:0*
T0*/
_output_shapes
:���������c"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������c:W S
/
_output_shapes
:���������c
 
_user_specified_nameinputs
�

�
D__inference_dense_8_layer_call_and_return_conditional_losses_1690712

inputs0
matmul_readvariableop_resource:c-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:c*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:���������`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������c: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������c
 
_user_specified_nameinputs
�
f
J__inference_CBlock_2_ReLu_layer_call_and_return_conditional_losses_1690485

inputs
identityP
Relu6Relu6inputs*
T0*/
_output_shapes
:���������	1c
IdentityIdentityRelu6:activations:0*
T0*/
_output_shapes
:���������	1"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������	1:W S
/
_output_shapes
:���������	1
 
_user_specified_nameinputs
�
s
W__inference_global_average_pooling2d_8_layer_call_and_return_conditional_losses_1689102

inputs
identityg
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      p
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:������������������^
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs"�
L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
K
input_layer<
serving_default_input_layer:0���������HX;
dense_80
StatefulPartitionedCall:0���������tensorflow/serving/predict:��
�
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
layer-6
layer_with_weights-5
layer-7
	layer_with_weights-6
	layer-8

layer-9
layer_with_weights-7
layer-10
layer_with_weights-8
layer-11
layer-12
layer-13
layer_with_weights-9
layer-14
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures"
_tf_keras_sequential
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
 bias
 !_jit_compiled_convolution_op"
_tf_keras_layer
�
"	variables
#trainable_variables
$regularization_losses
%	keras_api
&__call__
*'&call_and_return_all_conditional_losses
(depthwise_kernel
)pointwise_kernel
*bias
 +_jit_compiled_convolution_op"
_tf_keras_layer
�
,	variables
-trainable_variables
.regularization_losses
/	keras_api
0__call__
*1&call_and_return_all_conditional_losses
2axis
	3gamma
4beta
5moving_mean
6moving_variance"
_tf_keras_layer
�
7	variables
8trainable_variables
9regularization_losses
:	keras_api
;__call__
*<&call_and_return_all_conditional_losses"
_tf_keras_layer
�
=	variables
>trainable_variables
?regularization_losses
@	keras_api
A__call__
*B&call_and_return_all_conditional_losses
Cdepthwise_kernel
Dpointwise_kernel
Ebias
 F_jit_compiled_convolution_op"
_tf_keras_layer
�
G	variables
Htrainable_variables
Iregularization_losses
J	keras_api
K__call__
*L&call_and_return_all_conditional_losses
Maxis
	Ngamma
Obeta
Pmoving_mean
Qmoving_variance"
_tf_keras_layer
�
R	variables
Strainable_variables
Tregularization_losses
U	keras_api
V__call__
*W&call_and_return_all_conditional_losses"
_tf_keras_layer
�
X	variables
Ytrainable_variables
Zregularization_losses
[	keras_api
\__call__
*]&call_and_return_all_conditional_losses
^depthwise_kernel
_pointwise_kernel
`bias
 a_jit_compiled_convolution_op"
_tf_keras_layer
�
b	variables
ctrainable_variables
dregularization_losses
e	keras_api
f__call__
*g&call_and_return_all_conditional_losses
haxis
	igamma
jbeta
kmoving_mean
lmoving_variance"
_tf_keras_layer
�
m	variables
ntrainable_variables
oregularization_losses
p	keras_api
q__call__
*r&call_and_return_all_conditional_losses"
_tf_keras_layer
�
s	variables
ttrainable_variables
uregularization_losses
v	keras_api
w__call__
*x&call_and_return_all_conditional_losses
ydepthwise_kernel
zpointwise_kernel
{bias
 |_jit_compiled_convolution_op"
_tf_keras_layer
�
}	variables
~trainable_variables
regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis

�gamma
	�beta
�moving_mean
�moving_variance"
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
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
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
�
0
 1
(2
)3
*4
35
46
57
68
C9
D10
E11
N12
O13
P14
Q15
^16
_17
`18
i19
j20
k21
l22
y23
z24
{25
�26
�27
�28
�29
�30
�31"
trackable_list_wrapper
�
0
 1
(2
)3
*4
35
46
C7
D8
E9
N10
O11
^12
_13
`14
i15
j16
y17
z18
{19
�20
�21
�22
�23"
trackable_list_wrapper
 "
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
�
�trace_0
�trace_1
�trace_2
�trace_32�
F__inference_jojo_n8_r72x88_a0.7_b4_g2.2_strides2_layer_call_fn_1689475
F__inference_jojo_n8_r72x88_a0.7_b4_g2.2_strides2_layer_call_fn_1689627
F__inference_jojo_n8_r72x88_a0.7_b4_g2.2_strides2_layer_call_fn_1689955
F__inference_jojo_n8_r72x88_a0.7_b4_g2.2_strides2_layer_call_fn_1690024�
���
FullArgSpec)
args!�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1z�trace_2z�trace_3
�
�trace_0
�trace_1
�trace_2
�trace_32�
a__inference_jojo_n8_r72x88_a0.7_b4_g2.2_strides2_layer_call_and_return_conditional_losses_1689239
a__inference_jojo_n8_r72x88_a0.7_b4_g2.2_strides2_layer_call_and_return_conditional_losses_1689322
a__inference_jojo_n8_r72x88_a0.7_b4_g2.2_strides2_layer_call_and_return_conditional_losses_1690147
a__inference_jojo_n8_r72x88_a0.7_b4_g2.2_strides2_layer_call_and_return_conditional_losses_1690270�
���
FullArgSpec)
args!�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1z�trace_2z�trace_3
�B�
"__inference__wrapped_model_1688727input_layer"�
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
�
	�iter
�beta_1
�beta_2

�decay
�learning_ratem� m�(m�)m�*m�3m�4m�Cm�Dm�Em�Nm�Om�^m�_m�`m�im�jm�ym�zm�{m�	�m�	�m�	�m�	�m�v� v�(v�)v�*v�3v�4v�Cv�Dv�Ev�Nv�Ov�^v�_v�`v�iv�jv�yv�zv�{v�	�v�	�v�	�v�	�v�"
	optimizer
-
�serving_default"
signature_map
.
0
 1"
trackable_list_wrapper
.
0
 1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
'__inference_Conv0_layer_call_fn_1690279�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
�
�trace_02�
B__inference_Conv0_layer_call_and_return_conditional_losses_1690289�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
&:$2Conv0/kernel
:2
Conv0/bias
�2��
���
FullArgSpec
args�
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
annotations� *
 0
5
(0
)1
*2"
trackable_list_wrapper
5
(0
)1
*2"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
"	variables
#trainable_variables
$regularization_losses
&__call__
*'&call_and_return_all_conditional_losses
&'"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
4__inference_CBlock_1_SepConv2D_layer_call_fn_1690300�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
�
�trace_02�
O__inference_CBlock_1_SepConv2D_layer_call_and_return_conditional_losses_1690315�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
=:;2#CBlock_1_SepConv2D/depthwise_kernel
=:;2#CBlock_1_SepConv2D/pointwise_kernel
%:#2CBlock_1_SepConv2D/bias
�2��
���
FullArgSpec
args�
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
annotations� *
 0
<
30
41
52
63"
trackable_list_wrapper
.
30
41"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
,	variables
-trainable_variables
.regularization_losses
0__call__
*1&call_and_return_all_conditional_losses
&1"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
4__inference_CBlock_1_BatchNorm_layer_call_fn_1690328
4__inference_CBlock_1_BatchNorm_layer_call_fn_1690341�
���
FullArgSpec)
args!�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
O__inference_CBlock_1_BatchNorm_layer_call_and_return_conditional_losses_1690359
O__inference_CBlock_1_BatchNorm_layer_call_and_return_conditional_losses_1690377�
���
FullArgSpec)
args!�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
 "
trackable_list_wrapper
&:$2CBlock_1_BatchNorm/gamma
%:#2CBlock_1_BatchNorm/beta
.:, (2CBlock_1_BatchNorm/moving_mean
2:0 (2"CBlock_1_BatchNorm/moving_variance
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
7	variables
8trainable_variables
9regularization_losses
;__call__
*<&call_and_return_all_conditional_losses
&<"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
/__inference_CBlock_1_ReLu_layer_call_fn_1690382�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
�
�trace_02�
J__inference_CBlock_1_ReLu_layer_call_and_return_conditional_losses_1690387�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
5
C0
D1
E2"
trackable_list_wrapper
5
C0
D1
E2"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
=	variables
>trainable_variables
?regularization_losses
A__call__
*B&call_and_return_all_conditional_losses
&B"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
4__inference_CBlock_2_SepConv2D_layer_call_fn_1690398�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
�
�trace_02�
O__inference_CBlock_2_SepConv2D_layer_call_and_return_conditional_losses_1690413�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
=:;2#CBlock_2_SepConv2D/depthwise_kernel
=:;12#CBlock_2_SepConv2D/pointwise_kernel
%:#12CBlock_2_SepConv2D/bias
�2��
���
FullArgSpec
args�
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
annotations� *
 0
<
N0
O1
P2
Q3"
trackable_list_wrapper
.
N0
O1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
G	variables
Htrainable_variables
Iregularization_losses
K__call__
*L&call_and_return_all_conditional_losses
&L"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
4__inference_CBlock_2_BatchNorm_layer_call_fn_1690426
4__inference_CBlock_2_BatchNorm_layer_call_fn_1690439�
���
FullArgSpec)
args!�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
O__inference_CBlock_2_BatchNorm_layer_call_and_return_conditional_losses_1690457
O__inference_CBlock_2_BatchNorm_layer_call_and_return_conditional_losses_1690475�
���
FullArgSpec)
args!�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
 "
trackable_list_wrapper
&:$12CBlock_2_BatchNorm/gamma
%:#12CBlock_2_BatchNorm/beta
.:,1 (2CBlock_2_BatchNorm/moving_mean
2:01 (2"CBlock_2_BatchNorm/moving_variance
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
R	variables
Strainable_variables
Tregularization_losses
V__call__
*W&call_and_return_all_conditional_losses
&W"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
/__inference_CBlock_2_ReLu_layer_call_fn_1690480�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
�
�trace_02�
J__inference_CBlock_2_ReLu_layer_call_and_return_conditional_losses_1690485�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
5
^0
_1
`2"
trackable_list_wrapper
5
^0
_1
`2"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
X	variables
Ytrainable_variables
Zregularization_losses
\__call__
*]&call_and_return_all_conditional_losses
&]"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
4__inference_CBlock_3_SepConv2D_layer_call_fn_1690496�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
�
�trace_02�
O__inference_CBlock_3_SepConv2D_layer_call_and_return_conditional_losses_1690511�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
=:;12#CBlock_3_SepConv2D/depthwise_kernel
=:;1J2#CBlock_3_SepConv2D/pointwise_kernel
%:#J2CBlock_3_SepConv2D/bias
�2��
���
FullArgSpec
args�
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
annotations� *
 0
<
i0
j1
k2
l3"
trackable_list_wrapper
.
i0
j1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
b	variables
ctrainable_variables
dregularization_losses
f__call__
*g&call_and_return_all_conditional_losses
&g"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
4__inference_CBlock_3_BatchNorm_layer_call_fn_1690524
4__inference_CBlock_3_BatchNorm_layer_call_fn_1690537�
���
FullArgSpec)
args!�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
O__inference_CBlock_3_BatchNorm_layer_call_and_return_conditional_losses_1690555
O__inference_CBlock_3_BatchNorm_layer_call_and_return_conditional_losses_1690573�
���
FullArgSpec)
args!�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
 "
trackable_list_wrapper
&:$J2CBlock_3_BatchNorm/gamma
%:#J2CBlock_3_BatchNorm/beta
.:,J (2CBlock_3_BatchNorm/moving_mean
2:0J (2"CBlock_3_BatchNorm/moving_variance
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
m	variables
ntrainable_variables
oregularization_losses
q__call__
*r&call_and_return_all_conditional_losses
&r"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
/__inference_CBlock_3_ReLu_layer_call_fn_1690578�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
�
�trace_02�
J__inference_CBlock_3_ReLu_layer_call_and_return_conditional_losses_1690583�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
5
y0
z1
{2"
trackable_list_wrapper
5
y0
z1
{2"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
s	variables
ttrainable_variables
uregularization_losses
w__call__
*x&call_and_return_all_conditional_losses
&x"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
4__inference_CBlock_4_SepConv2D_layer_call_fn_1690594�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
�
�trace_02�
O__inference_CBlock_4_SepConv2D_layer_call_and_return_conditional_losses_1690609�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
=:;J2#CBlock_4_SepConv2D/depthwise_kernel
=:;Jc2#CBlock_4_SepConv2D/pointwise_kernel
%:#c2CBlock_4_SepConv2D/bias
�2��
���
FullArgSpec
args�
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
annotations� *
 0
@
�0
�1
�2
�3"
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
}	variables
~trainable_variables
regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
4__inference_CBlock_4_BatchNorm_layer_call_fn_1690622
4__inference_CBlock_4_BatchNorm_layer_call_fn_1690635�
���
FullArgSpec)
args!�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
O__inference_CBlock_4_BatchNorm_layer_call_and_return_conditional_losses_1690653
O__inference_CBlock_4_BatchNorm_layer_call_and_return_conditional_losses_1690671�
���
FullArgSpec)
args!�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
 "
trackable_list_wrapper
&:$c2CBlock_4_BatchNorm/gamma
%:#c2CBlock_4_BatchNorm/beta
.:,c (2CBlock_4_BatchNorm/moving_mean
2:0c (2"CBlock_4_BatchNorm/moving_variance
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
/__inference_CBlock_4_ReLu_layer_call_fn_1690676�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
�
�trace_02�
J__inference_CBlock_4_ReLu_layer_call_and_return_conditional_losses_1690681�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
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
<__inference_global_average_pooling2d_8_layer_call_fn_1690686�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
�
�trace_02�
W__inference_global_average_pooling2d_8_layer_call_and_return_conditional_losses_1690692�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
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
)__inference_dense_8_layer_call_fn_1690701�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
�
�trace_02�
D__inference_dense_8_layer_call_and_return_conditional_losses_1690712�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
 :c2dense_8/kernel
:2dense_8/bias
Z
50
61
P2
Q3
k4
l5
�6
�7"
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
�B�
F__inference_jojo_n8_r72x88_a0.7_b4_g2.2_strides2_layer_call_fn_1689475input_layer"�
���
FullArgSpec)
args!�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_jojo_n8_r72x88_a0.7_b4_g2.2_strides2_layer_call_fn_1689627input_layer"�
���
FullArgSpec)
args!�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_jojo_n8_r72x88_a0.7_b4_g2.2_strides2_layer_call_fn_1689955inputs"�
���
FullArgSpec)
args!�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_jojo_n8_r72x88_a0.7_b4_g2.2_strides2_layer_call_fn_1690024inputs"�
���
FullArgSpec)
args!�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
a__inference_jojo_n8_r72x88_a0.7_b4_g2.2_strides2_layer_call_and_return_conditional_losses_1689239input_layer"�
���
FullArgSpec)
args!�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
a__inference_jojo_n8_r72x88_a0.7_b4_g2.2_strides2_layer_call_and_return_conditional_losses_1689322input_layer"�
���
FullArgSpec)
args!�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
a__inference_jojo_n8_r72x88_a0.7_b4_g2.2_strides2_layer_call_and_return_conditional_losses_1690147inputs"�
���
FullArgSpec)
args!�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
a__inference_jojo_n8_r72x88_a0.7_b4_g2.2_strides2_layer_call_and_return_conditional_losses_1690270inputs"�
���
FullArgSpec)
args!�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
�B�
%__inference_signature_wrapper_1689886input_layer"�
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
'__inference_Conv0_layer_call_fn_1690279inputs"�
���
FullArgSpec
args�

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
annotations� *
 
�B�
B__inference_Conv0_layer_call_and_return_conditional_losses_1690289inputs"�
���
FullArgSpec
args�

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
annotations� *
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
4__inference_CBlock_1_SepConv2D_layer_call_fn_1690300inputs"�
���
FullArgSpec
args�

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
annotations� *
 
�B�
O__inference_CBlock_1_SepConv2D_layer_call_and_return_conditional_losses_1690315inputs"�
���
FullArgSpec
args�

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
annotations� *
 
.
50
61"
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
4__inference_CBlock_1_BatchNorm_layer_call_fn_1690328inputs"�
���
FullArgSpec)
args!�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
4__inference_CBlock_1_BatchNorm_layer_call_fn_1690341inputs"�
���
FullArgSpec)
args!�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
O__inference_CBlock_1_BatchNorm_layer_call_and_return_conditional_losses_1690359inputs"�
���
FullArgSpec)
args!�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
O__inference_CBlock_1_BatchNorm_layer_call_and_return_conditional_losses_1690377inputs"�
���
FullArgSpec)
args!�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
/__inference_CBlock_1_ReLu_layer_call_fn_1690382inputs"�
���
FullArgSpec
args�

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
annotations� *
 
�B�
J__inference_CBlock_1_ReLu_layer_call_and_return_conditional_losses_1690387inputs"�
���
FullArgSpec
args�

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
annotations� *
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
4__inference_CBlock_2_SepConv2D_layer_call_fn_1690398inputs"�
���
FullArgSpec
args�

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
annotations� *
 
�B�
O__inference_CBlock_2_SepConv2D_layer_call_and_return_conditional_losses_1690413inputs"�
���
FullArgSpec
args�

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
annotations� *
 
.
P0
Q1"
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
4__inference_CBlock_2_BatchNorm_layer_call_fn_1690426inputs"�
���
FullArgSpec)
args!�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
4__inference_CBlock_2_BatchNorm_layer_call_fn_1690439inputs"�
���
FullArgSpec)
args!�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
O__inference_CBlock_2_BatchNorm_layer_call_and_return_conditional_losses_1690457inputs"�
���
FullArgSpec)
args!�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
O__inference_CBlock_2_BatchNorm_layer_call_and_return_conditional_losses_1690475inputs"�
���
FullArgSpec)
args!�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
/__inference_CBlock_2_ReLu_layer_call_fn_1690480inputs"�
���
FullArgSpec
args�

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
annotations� *
 
�B�
J__inference_CBlock_2_ReLu_layer_call_and_return_conditional_losses_1690485inputs"�
���
FullArgSpec
args�

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
annotations� *
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
4__inference_CBlock_3_SepConv2D_layer_call_fn_1690496inputs"�
���
FullArgSpec
args�

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
annotations� *
 
�B�
O__inference_CBlock_3_SepConv2D_layer_call_and_return_conditional_losses_1690511inputs"�
���
FullArgSpec
args�

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
annotations� *
 
.
k0
l1"
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
4__inference_CBlock_3_BatchNorm_layer_call_fn_1690524inputs"�
���
FullArgSpec)
args!�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
4__inference_CBlock_3_BatchNorm_layer_call_fn_1690537inputs"�
���
FullArgSpec)
args!�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
O__inference_CBlock_3_BatchNorm_layer_call_and_return_conditional_losses_1690555inputs"�
���
FullArgSpec)
args!�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
O__inference_CBlock_3_BatchNorm_layer_call_and_return_conditional_losses_1690573inputs"�
���
FullArgSpec)
args!�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
/__inference_CBlock_3_ReLu_layer_call_fn_1690578inputs"�
���
FullArgSpec
args�

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
annotations� *
 
�B�
J__inference_CBlock_3_ReLu_layer_call_and_return_conditional_losses_1690583inputs"�
���
FullArgSpec
args�

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
annotations� *
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
4__inference_CBlock_4_SepConv2D_layer_call_fn_1690594inputs"�
���
FullArgSpec
args�

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
annotations� *
 
�B�
O__inference_CBlock_4_SepConv2D_layer_call_and_return_conditional_losses_1690609inputs"�
���
FullArgSpec
args�

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
annotations� *
 
0
�0
�1"
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
4__inference_CBlock_4_BatchNorm_layer_call_fn_1690622inputs"�
���
FullArgSpec)
args!�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
4__inference_CBlock_4_BatchNorm_layer_call_fn_1690635inputs"�
���
FullArgSpec)
args!�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
O__inference_CBlock_4_BatchNorm_layer_call_and_return_conditional_losses_1690653inputs"�
���
FullArgSpec)
args!�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
O__inference_CBlock_4_BatchNorm_layer_call_and_return_conditional_losses_1690671inputs"�
���
FullArgSpec)
args!�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
/__inference_CBlock_4_ReLu_layer_call_fn_1690676inputs"�
���
FullArgSpec
args�

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
annotations� *
 
�B�
J__inference_CBlock_4_ReLu_layer_call_and_return_conditional_losses_1690681inputs"�
���
FullArgSpec
args�

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
annotations� *
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
<__inference_global_average_pooling2d_8_layer_call_fn_1690686inputs"�
���
FullArgSpec
args�

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
annotations� *
 
�B�
W__inference_global_average_pooling2d_8_layer_call_and_return_conditional_losses_1690692inputs"�
���
FullArgSpec
args�

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
annotations� *
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
)__inference_dense_8_layer_call_fn_1690701inputs"�
���
FullArgSpec
args�

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
annotations� *
 
�B�
D__inference_dense_8_layer_call_and_return_conditional_losses_1690712inputs"�
���
FullArgSpec
args�

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
annotations� *
 
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
+:)2Adam/Conv0/kernel/m
:2Adam/Conv0/bias/m
B:@2*Adam/CBlock_1_SepConv2D/depthwise_kernel/m
B:@2*Adam/CBlock_1_SepConv2D/pointwise_kernel/m
*:(2Adam/CBlock_1_SepConv2D/bias/m
+:)2Adam/CBlock_1_BatchNorm/gamma/m
*:(2Adam/CBlock_1_BatchNorm/beta/m
B:@2*Adam/CBlock_2_SepConv2D/depthwise_kernel/m
B:@12*Adam/CBlock_2_SepConv2D/pointwise_kernel/m
*:(12Adam/CBlock_2_SepConv2D/bias/m
+:)12Adam/CBlock_2_BatchNorm/gamma/m
*:(12Adam/CBlock_2_BatchNorm/beta/m
B:@12*Adam/CBlock_3_SepConv2D/depthwise_kernel/m
B:@1J2*Adam/CBlock_3_SepConv2D/pointwise_kernel/m
*:(J2Adam/CBlock_3_SepConv2D/bias/m
+:)J2Adam/CBlock_3_BatchNorm/gamma/m
*:(J2Adam/CBlock_3_BatchNorm/beta/m
B:@J2*Adam/CBlock_4_SepConv2D/depthwise_kernel/m
B:@Jc2*Adam/CBlock_4_SepConv2D/pointwise_kernel/m
*:(c2Adam/CBlock_4_SepConv2D/bias/m
+:)c2Adam/CBlock_4_BatchNorm/gamma/m
*:(c2Adam/CBlock_4_BatchNorm/beta/m
%:#c2Adam/dense_8/kernel/m
:2Adam/dense_8/bias/m
+:)2Adam/Conv0/kernel/v
:2Adam/Conv0/bias/v
B:@2*Adam/CBlock_1_SepConv2D/depthwise_kernel/v
B:@2*Adam/CBlock_1_SepConv2D/pointwise_kernel/v
*:(2Adam/CBlock_1_SepConv2D/bias/v
+:)2Adam/CBlock_1_BatchNorm/gamma/v
*:(2Adam/CBlock_1_BatchNorm/beta/v
B:@2*Adam/CBlock_2_SepConv2D/depthwise_kernel/v
B:@12*Adam/CBlock_2_SepConv2D/pointwise_kernel/v
*:(12Adam/CBlock_2_SepConv2D/bias/v
+:)12Adam/CBlock_2_BatchNorm/gamma/v
*:(12Adam/CBlock_2_BatchNorm/beta/v
B:@12*Adam/CBlock_3_SepConv2D/depthwise_kernel/v
B:@1J2*Adam/CBlock_3_SepConv2D/pointwise_kernel/v
*:(J2Adam/CBlock_3_SepConv2D/bias/v
+:)J2Adam/CBlock_3_BatchNorm/gamma/v
*:(J2Adam/CBlock_3_BatchNorm/beta/v
B:@J2*Adam/CBlock_4_SepConv2D/depthwise_kernel/v
B:@Jc2*Adam/CBlock_4_SepConv2D/pointwise_kernel/v
*:(c2Adam/CBlock_4_SepConv2D/bias/v
+:)c2Adam/CBlock_4_BatchNorm/gamma/v
*:(c2Adam/CBlock_4_BatchNorm/beta/v
%:#c2Adam/dense_8/kernel/v
:2Adam/dense_8/bias/v�
O__inference_CBlock_1_BatchNorm_layer_call_and_return_conditional_losses_1690359�3456Q�N
G�D
:�7
inputs+���������������������������
p

 
� "F�C
<�9
tensor_0+���������������������������
� �
O__inference_CBlock_1_BatchNorm_layer_call_and_return_conditional_losses_1690377�3456Q�N
G�D
:�7
inputs+���������������������������
p 

 
� "F�C
<�9
tensor_0+���������������������������
� �
4__inference_CBlock_1_BatchNorm_layer_call_fn_1690328�3456Q�N
G�D
:�7
inputs+���������������������������
p

 
� ";�8
unknown+����������������������������
4__inference_CBlock_1_BatchNorm_layer_call_fn_1690341�3456Q�N
G�D
:�7
inputs+���������������������������
p 

 
� ";�8
unknown+����������������������������
J__inference_CBlock_1_ReLu_layer_call_and_return_conditional_losses_1690387o7�4
-�*
(�%
inputs���������
� "4�1
*�'
tensor_0���������
� �
/__inference_CBlock_1_ReLu_layer_call_fn_1690382d7�4
-�*
(�%
inputs���������
� ")�&
unknown����������
O__inference_CBlock_1_SepConv2D_layer_call_and_return_conditional_losses_1690315�()*I�F
?�<
:�7
inputs+���������������������������
� "F�C
<�9
tensor_0+���������������������������
� �
4__inference_CBlock_1_SepConv2D_layer_call_fn_1690300�()*I�F
?�<
:�7
inputs+���������������������������
� ";�8
unknown+����������������������������
O__inference_CBlock_2_BatchNorm_layer_call_and_return_conditional_losses_1690457�NOPQQ�N
G�D
:�7
inputs+���������������������������1
p

 
� "F�C
<�9
tensor_0+���������������������������1
� �
O__inference_CBlock_2_BatchNorm_layer_call_and_return_conditional_losses_1690475�NOPQQ�N
G�D
:�7
inputs+���������������������������1
p 

 
� "F�C
<�9
tensor_0+���������������������������1
� �
4__inference_CBlock_2_BatchNorm_layer_call_fn_1690426�NOPQQ�N
G�D
:�7
inputs+���������������������������1
p

 
� ";�8
unknown+���������������������������1�
4__inference_CBlock_2_BatchNorm_layer_call_fn_1690439�NOPQQ�N
G�D
:�7
inputs+���������������������������1
p 

 
� ";�8
unknown+���������������������������1�
J__inference_CBlock_2_ReLu_layer_call_and_return_conditional_losses_1690485o7�4
-�*
(�%
inputs���������	1
� "4�1
*�'
tensor_0���������	1
� �
/__inference_CBlock_2_ReLu_layer_call_fn_1690480d7�4
-�*
(�%
inputs���������	1
� ")�&
unknown���������	1�
O__inference_CBlock_2_SepConv2D_layer_call_and_return_conditional_losses_1690413�CDEI�F
?�<
:�7
inputs+���������������������������
� "F�C
<�9
tensor_0+���������������������������1
� �
4__inference_CBlock_2_SepConv2D_layer_call_fn_1690398�CDEI�F
?�<
:�7
inputs+���������������������������
� ";�8
unknown+���������������������������1�
O__inference_CBlock_3_BatchNorm_layer_call_and_return_conditional_losses_1690555�ijklQ�N
G�D
:�7
inputs+���������������������������J
p

 
� "F�C
<�9
tensor_0+���������������������������J
� �
O__inference_CBlock_3_BatchNorm_layer_call_and_return_conditional_losses_1690573�ijklQ�N
G�D
:�7
inputs+���������������������������J
p 

 
� "F�C
<�9
tensor_0+���������������������������J
� �
4__inference_CBlock_3_BatchNorm_layer_call_fn_1690524�ijklQ�N
G�D
:�7
inputs+���������������������������J
p

 
� ";�8
unknown+���������������������������J�
4__inference_CBlock_3_BatchNorm_layer_call_fn_1690537�ijklQ�N
G�D
:�7
inputs+���������������������������J
p 

 
� ";�8
unknown+���������������������������J�
J__inference_CBlock_3_ReLu_layer_call_and_return_conditional_losses_1690583o7�4
-�*
(�%
inputs���������J
� "4�1
*�'
tensor_0���������J
� �
/__inference_CBlock_3_ReLu_layer_call_fn_1690578d7�4
-�*
(�%
inputs���������J
� ")�&
unknown���������J�
O__inference_CBlock_3_SepConv2D_layer_call_and_return_conditional_losses_1690511�^_`I�F
?�<
:�7
inputs+���������������������������1
� "F�C
<�9
tensor_0+���������������������������J
� �
4__inference_CBlock_3_SepConv2D_layer_call_fn_1690496�^_`I�F
?�<
:�7
inputs+���������������������������1
� ";�8
unknown+���������������������������J�
O__inference_CBlock_4_BatchNorm_layer_call_and_return_conditional_losses_1690653�����Q�N
G�D
:�7
inputs+���������������������������c
p

 
� "F�C
<�9
tensor_0+���������������������������c
� �
O__inference_CBlock_4_BatchNorm_layer_call_and_return_conditional_losses_1690671�����Q�N
G�D
:�7
inputs+���������������������������c
p 

 
� "F�C
<�9
tensor_0+���������������������������c
� �
4__inference_CBlock_4_BatchNorm_layer_call_fn_1690622�����Q�N
G�D
:�7
inputs+���������������������������c
p

 
� ";�8
unknown+���������������������������c�
4__inference_CBlock_4_BatchNorm_layer_call_fn_1690635�����Q�N
G�D
:�7
inputs+���������������������������c
p 

 
� ";�8
unknown+���������������������������c�
J__inference_CBlock_4_ReLu_layer_call_and_return_conditional_losses_1690681o7�4
-�*
(�%
inputs���������c
� "4�1
*�'
tensor_0���������c
� �
/__inference_CBlock_4_ReLu_layer_call_fn_1690676d7�4
-�*
(�%
inputs���������c
� ")�&
unknown���������c�
O__inference_CBlock_4_SepConv2D_layer_call_and_return_conditional_losses_1690609�yz{I�F
?�<
:�7
inputs+���������������������������J
� "F�C
<�9
tensor_0+���������������������������c
� �
4__inference_CBlock_4_SepConv2D_layer_call_fn_1690594�yz{I�F
?�<
:�7
inputs+���������������������������J
� ";�8
unknown+���������������������������c�
B__inference_Conv0_layer_call_and_return_conditional_losses_1690289s 7�4
-�*
(�%
inputs���������HX
� "4�1
*�'
tensor_0���������$,
� �
'__inference_Conv0_layer_call_fn_1690279h 7�4
-�*
(�%
inputs���������HX
� ")�&
unknown���������$,�
"__inference__wrapped_model_1688727�& ()*3456CDENOPQ^_`ijklyz{������<�9
2�/
-�*
input_layer���������HX
� "1�.
,
dense_8!�
dense_8����������
D__inference_dense_8_layer_call_and_return_conditional_losses_1690712e��/�,
%�"
 �
inputs���������c
� ",�)
"�
tensor_0���������
� �
)__inference_dense_8_layer_call_fn_1690701Z��/�,
%�"
 �
inputs���������c
� "!�
unknown����������
W__inference_global_average_pooling2d_8_layer_call_and_return_conditional_losses_1690692�R�O
H�E
C�@
inputs4������������������������������������
� "5�2
+�(
tensor_0������������������
� �
<__inference_global_average_pooling2d_8_layer_call_fn_1690686�R�O
H�E
C�@
inputs4������������������������������������
� "*�'
unknown�������������������
a__inference_jojo_n8_r72x88_a0.7_b4_g2.2_strides2_layer_call_and_return_conditional_losses_1689239�& ()*3456CDENOPQ^_`ijklyz{������D�A
:�7
-�*
input_layer���������HX
p

 
� ",�)
"�
tensor_0���������
� �
a__inference_jojo_n8_r72x88_a0.7_b4_g2.2_strides2_layer_call_and_return_conditional_losses_1689322�& ()*3456CDENOPQ^_`ijklyz{������D�A
:�7
-�*
input_layer���������HX
p 

 
� ",�)
"�
tensor_0���������
� �
a__inference_jojo_n8_r72x88_a0.7_b4_g2.2_strides2_layer_call_and_return_conditional_losses_1690147�& ()*3456CDENOPQ^_`ijklyz{������?�<
5�2
(�%
inputs���������HX
p

 
� ",�)
"�
tensor_0���������
� �
a__inference_jojo_n8_r72x88_a0.7_b4_g2.2_strides2_layer_call_and_return_conditional_losses_1690270�& ()*3456CDENOPQ^_`ijklyz{������?�<
5�2
(�%
inputs���������HX
p 

 
� ",�)
"�
tensor_0���������
� �
F__inference_jojo_n8_r72x88_a0.7_b4_g2.2_strides2_layer_call_fn_1689475�& ()*3456CDENOPQ^_`ijklyz{������D�A
:�7
-�*
input_layer���������HX
p

 
� "!�
unknown����������
F__inference_jojo_n8_r72x88_a0.7_b4_g2.2_strides2_layer_call_fn_1689627�& ()*3456CDENOPQ^_`ijklyz{������D�A
:�7
-�*
input_layer���������HX
p 

 
� "!�
unknown����������
F__inference_jojo_n8_r72x88_a0.7_b4_g2.2_strides2_layer_call_fn_1689955�& ()*3456CDENOPQ^_`ijklyz{������?�<
5�2
(�%
inputs���������HX
p

 
� "!�
unknown����������
F__inference_jojo_n8_r72x88_a0.7_b4_g2.2_strides2_layer_call_fn_1690024�& ()*3456CDENOPQ^_`ijklyz{������?�<
5�2
(�%
inputs���������HX
p 

 
� "!�
unknown����������
%__inference_signature_wrapper_1689886�& ()*3456CDENOPQ^_`ijklyz{������K�H
� 
A�>
<
input_layer-�*
input_layer���������HX"1�.
,
dense_8!�
dense_8���������