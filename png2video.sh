#EXAMPLE: ./png2video.sh $HOME/DockerVolume/seg4art/data/scenes/tiktok2

INPUT_DIR=$1

#put the images together to make the MP4 video
ffmpeg -y -r 25 -i $INPUT_DIR/out_pngs/%05d.png -c:v libx264 -vf fps=25 -pix_fmt yuv420p $INPUT_DIR/out.mp4
