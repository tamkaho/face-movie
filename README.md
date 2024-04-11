# face-movie

<img src="https://github.com/andrewdcampbell/face-movie/blob/master/demos/demo.gif" width="900">

Forked from <https://github.com/andrewdcampbell/face-movie> with the following improvements. Primarily:

- Replaced dlib with mediapipe
- For images where faces aren't detected, allows the user to click twice to select the eye coordinates. Saves a json of eye coords for images where face detection fails. If multiple faces are detected, the eye coordinate selection is used to pick the face.
- Runs on Windows and Unix systems.
- Some bug fixes with frame rates and morph.
- Option to output landmarks.
- Option to add text to each frame.
- sliding window time average of faces

Create a video warp sequence of human faces. Can be used, for example, to create a time-lapse video showing someone's face change over time. See demos [here](https://www.youtube.com/watch?v=sbHCar2T-e0) and [here](https://www.youtube.com/watch?v=mmz0FE6lT5A).

Supported on Python 3 and OpenCV 3+.

## Requirements

- OpenCV
  - For conda users, run `conda install -c conda-forge opencv`.
- Face Recognition
  - Run `pip install mediapipe`.
- ffmpeg
  - For conda users, run `conda install -c conda-forge ffmpeg`.
- scipy
- numpy
- matplotlib
- pillow

## Installation

1. Clone the repo.

```bash
git clone https://github.com/tamkaho/face-movie
```

## Creating a face movie - reccomended workflow

1. Make a directory `<FACE_MOVIE_DIR>` in the root directory of the repo with the desired face images. The images must feature a clear frontal view of the desired face (other faces can be present too). The image filenames must be in lexicographic order of the order in which they are to appear in the video.

2. Create a directory `<ALIGN_OUTPUT>`. Then align the faces in the images with  &
  
      ```bash
      python face-movie/align.py -images <FACE_MOVIE_DIR> -target <BASE_IMAGE>
                                 [-overlay] [-border <BORDER>] -outdir <ALIGN_OUTPUT>  
      ```

    The output will be saved to the provided `<ALIGN_OUTPUT>` directory. BASE_IMAGE is the image to which all other images will be aligned to. It should represent the "typical" image of all your images - it will determine the output dimensions and facial position.  

    The optional `-overlay` flag places subsequent images on top of each other (recommended). The optional `-border <BORDER>` argument adds a white border `<BORDER>` pixels across to all the images for aesthetics. I think around 5 pixels looks good. Create a dir named "landmark" and add `-landmark` if you want to output landmark files. If some of the faces fail to align properly, delete those images, add `-manual_eye` and run the command again.

    If the code fails to detect exactly one face, a window will appear. Click on the two eyes and this will either pick a face (if multiple faces are detected) or the face will be aligned using the eyes only (if no faces are detected).

3. Morph the sequence with

      ```bash
      python face-movie/main.py -morph -images <ALIGN_OUTPUT> -tf <TOTAL_FRAMES>
                                -fps <FPS> -out <OUTPUT_NAME>.mp4
      ```

    This will create a video `OUTPUT_NAME.mp4` in the root directory with the desired parameters. Note that `TOTAL_FRAMES`, and `FPS` are an integers. Optionally, add `-text_prefix` followed by some text to write some text with the image number at the bottom of each frame (use `-txt_dist_bottom` to adjust the y position of the text).

    Adding `-running_avg` followed by a number `n` will create timelapse with a sliding window weighted moving average across `2*n+1` days. For this to work the input frames must be prefixed `YYYYMMDD`. The step size of the sliding window is defined using `-day_step`: this can any float values to control the smoothness of the movie. I'm using to visualise the growth of my child by taking a photo of him each day. This helps to reduce flickering in timelapse. Time average images tend to have a boring background - the option `-face_only_running_avg` allows the moving average to be applied to the face region only while keeping the background interesting. So far I have managed to take photos of my child everyday but the code should still work should we manage miss a few days of photos.

4. Creating the video
To create a video straight from align (step 2) or from the frames output from `running_avg` in step 3, use `timelapse.py`.

5. (Optional) Add music with
  
      ```bash
      ffmpeg -i <OUTPUT_NAME>.mp4 -i <AUDIO_TRACK> -map 0:v -map 1:a -c copy
             -shortest -pix_fmt yuv420p <NEW_OUTPUT_NAME>.mov
      ```

    The images used to produce the first demo video are included in `demos/daniel`. Step 2 should produce something like the aligned images shown in `demos/daniel_aligned`. Step 3 should produce something like the video `daniel.mp4`.

### Notes

- If you have photos that are already the same size and all of the faces are more or less aligned already (e.g. roster photos or yearbook photos), you can skip steps 1 and 2 and just use the directory of photos as your aligned images.

- If you want to add some photos to the end of an existing morph sequence (and avoid rendering the entire video again), you can make a temporary directory containing the last aligned image in the movie and aligned images you want to append. Make a morph sequence out of just those images, making sure to use the same frame rate. Now concatenate them with

    ```bash
    ffmpeg -safe 0 -f concat -i list.txt -c copy merged.mp4
    ```

  where `list.txt` is of the form

    ```bash
    file 'path/to/video1.mp4'
    file 'path/to/video2.mp4'
    ...
    ```

- For adding music longer than the video, you can fade the audio out with

    ```bash
    VIDEO="<PATH_TO_VIDEO>"
    AUDIO="<PATH_TO_AUDIO>" # can also drag and drop audio track to terminal (don't use quotes)
    FADE="<FADE_DURATION_IN_SECONDS>"

    VIDEO_LENGTH="$(ffprobe -v error -select_streams v:0 -show_entries stream=duration -of csv=p=0 $VIDEO)"
    ffmpeg -i $VIDEO -i "$AUDIO" -filter_complex "[1:a]afade=t=out:st=$(bc <<< "$VIDEO_LENGTH-$FADE"):d=$FADE[a]" -map 0:v:0 -map "[a]" -c:v copy -c:a aac -shortest with_audio.mp4
    ```

## Averaging Faces

You can also use the code to create a face average. Follow the same steps 1) - 2) as above. You probably don't want to overlay images or use a border, however. Then run

  ```bash
  python face-movie/main.py -average -images <ALIGN_OUTPUT> -out <OUTPUT_NAME>.jpg
  ```

A small face dataset is included in the demos directory.

<img src="https://github.com/andrewdcampbell/face-movie/blob/master/demos/face_dataset/male_faces.png" width="500">
<img src="https://github.com/andrewdcampbell/face-movie/blob/master/demos/face_dataset/female_faces.png" width="500">

The computed average male and female face are shown below.

<img src="https://github.com/andrewdcampbell/face-movie/blob/master/demos/male_avg.jpg" width="250"> <img src="https://github.com/andrewdcampbell/face-movie/blob/master/demos/female_avg.jpg" width="250">

## Acknowledgements

- Facial landmark and image alignment code adapted from <https://matthewearl.github.io/2015/07/28/switching-eds-with-python/>.
- ffmpeg command adapted from <https://github.com/kubricio/face-morphing>.
- Affine transform code adapted from <https://www.learnopencv.com/face-morph-using-opencv-cpp-python/>.
