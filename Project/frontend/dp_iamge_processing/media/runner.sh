#!/bin/bash

TASK_ID="$1"
TRAIN="$2"
IMAGE_SOURCE="$3"
INPUT_BASIC="$4"
OUTPUT_BASIC="$5"
LAYERS_BASIC="$6"
INPUT_COMPLEX="$7"
OUTPUT_COMPLEX="$8"
LAYERS_COMPLEX="$9"
DESIRED_ERROR="${10}"
MAX_EPOCHS="${11}"
HIDDEN_NEURON1="${12}"
HIDDEN_NEURON2="${13}"
ACTIVATION_FUNC_HIDDEN="${14}"
ACTIVATION_FUNC_OUTPUT="${15}"
SIZE_OF_BLOCK_ORIANTATION="${16}"
SIZE_OF_BLOCK_GABOR="${17}"
SIGMA="${18}"
LAMBDA="${19}"
GAMMA="${20}"


cd /home/gregor/projects/dp_image_processing/dp_iamge_processing/media/tasks/

mkdir $TASK_ID
cd $TASK_ID
mkdir output
cd ..
cd ..

echo "RUNNING" > ./tasks/$TASK_ID/status
echo "Start" >> ./tasks/$TASK_ID/outputs

./DP2016 $TASK_ID $TRAIN $IMAGE_SOURCE $INPUT_BASIC $OUTPUT_BASIC $LAYERS_BASIC $INPUT_COMPLEX $OUTPUT_COMPLEX $LAYERS_COMPLEX $DESIRED_ERROR $MAX_EPOCHS $HIDDEN_NEURON1 $HIDDEN_NEURON2 $ACTIVATION_FUNC_HIDDEN $ACTIVATION_FUNC_OUTPUT $SIZE_OF_BLOCK_ORIANTATION $SIZE_OF_BLOCK_GABOR $SIGMA $LAMBDA $GAMMA >>./tasks/$TASK_ID/report.txt

echo "Finish" >> ./tasks/$TASK_ID/outputs

echo "COMPLETED" > ./tasks/$TASK_ID/status


exit 0
