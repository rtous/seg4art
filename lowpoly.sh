#EXAMPLE: ./lowpoly.sh green_woman3 1
#EXAMPLE: ./lowpoly.sh man_walk_1_part1 0

SCENE_NAME=$1
ADDFACE=$2
python lowpoly_last.py $SCENE_NAME $ADDFACE