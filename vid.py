import cv2
from moviepy.editor import VideoFileClip
from pygame import mixer

def get_audio(file):
    video = VideoFileClip(file + ".mp4")
    audio = video.audio
    audio.write_audiofile(file + ".mp3")

def play_video(video_path):
    mixer.init()  # Initialize the mixer
    mixer.music.load(video_path + ".mp3")  # Load the audio file
    mixer.music.play()  # Start playing the audio

    video = cv2.VideoCapture(video_path + ".mp4")

    while True:
        grabbed, frame = video.read()
        frame = cv2.resize(frame, (800, 500))

        if not grabbed:
            print("End of video")
            break

        if cv2.waitKey(100) & 0xFF == ord("q"):
            break

        cv2.imshow("Video", frame)

    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = "dog_cat"
    get_audio(video_path)
    play_video(video_path)
