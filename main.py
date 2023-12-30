import streamlit as st
import cv2
import numpy as np
from pytube import YouTube
from moviepy.editor import VideoFileClip
from collections import deque
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model

# Set the dimensions to which each video frame will be resized in our dataset.
IMAGE_HEIGHT, IMAGE_WIDTH = 64, 64

# Define the number of frames per video sequence fed to the model.
SEQUENCE_LENGTH = 20

# Designate the directory containing the UCF50 dataset. 
DATASET_DIR = "dataset/UCF50"

# Specify the list that holds the names of the classes intended for training. Feel free to select any desired set of classes.
CLASSES_LIST = ["WalkingWithDog", "TaiChi", "Swing", "HorseRace", "Basketball", "PushUps"]


@st.cache_resource
def load_lrcn_model():
    return load_model("LRCN_model_Date_Time_2023_12_29_12_15_56___Loss_0.46385160088539124___Accuracy_0.9125683307647705.h5")

LRCN_model = load_lrcn_model()

def frames_extraction(video_path):
    '''
    This function extracts the necessary frames from a video after resizing and normalization.
    Parameters:
        video_path: The disk path of the video from which frames are to be extracted.
    Returns:
        frames_list: A list containing the resized and normalized frames of the video.
    '''
    # Initialize an empty list to store frames.
    frames_list = []
    
    # Open the video file using OpenCV's VideoCapture.
    video_reader = cv2.VideoCapture(video_path)
    
    # Obtain the total number of frames in the video.
    video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate the window for skipping frames based on the desired sequence length.
    skip_frames_window = max(int(video_frames_count / SEQUENCE_LENGTH), 1)
    
    # Iterate through the frames to extract the required sequence.
    for frame_counter in range(SEQUENCE_LENGTH):
        # Set the position to read a specific frame based on the skip_frames_window.
        video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter * skip_frames_window)
        
        # Read the frame from the video.
        success, frame = video_reader.read()
        
        # Check for read success; if not, print an error message and exit the loop.
        if not success:
            print("Error reading video frames")
            break
        
        # Resize the frame to the specified dimensions.
        resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
        
        # Normalize the pixel values to the range [0, 1].
        normalized_frame = resized_frame / 255
        
        # Append the normalized frame to the frames_list.
        frames_list.append(normalized_frame)
    
    # Release the video reader.
    video_reader.release()
    
    # Return the list containing the resized and normalized frames.
    return frames_list

def download_youtube_videos(youtube_video_url, output_directory):
    '''
    This function downloads the YouTube video specified by the provided URL.
    Args:
        youtube_video_url: URL of the video to be downloaded.
        output_directory: The directory path where the downloaded video will be stored.
    Returns:
        title: The title of the downloaded YouTube video.
    '''
    # Create a YouTube object for the YouTube video.
    youtube_video = YouTube(youtube_video_url)
    
    # Obtain the title of the video.
    title = youtube_video.title
    
    # Retrieve the stream with the highest resolution.
    video_stream = youtube_video.streams.get_highest_resolution()
    
    # Define the output file path using the video title.
    output_file_path = f'{output_directory}/{title}.mp4'
    
    # Download the video to the specified output directory.
    video_stream.download(output_directory)
    
    # Return the title of the downloaded YouTube video.
    return title
    
def predict_on_video(video_file_path, output_file_path, SEQUENCE_LENGTH):
    '''
    This function conducts action recognition on a video using the LRCN model.
    Parameters:
    video_file_path: Path to the video on disk for action recognition.
    output_file_path: Path to store the output video with overlaid predicted actions.
    SEQUENCE_LENGTH: Fixed number of frames forming a sequence input for the model.
    '''
    # Open the video file for reading.
    video_reader = cv2.VideoCapture(video_file_path)
    
    # Get the original video dimensions.
    original_video_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_video_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Open a video file for writing with the same dimensions and frame rate.
    video_writer = cv2.VideoWriter(output_file_path, 
                                   cv2.VideoWriter_fourcc('H', '2', '6', '4'), 
                                   video_reader.get(cv2.CAP_PROP_FPS), 
                                   (original_video_width, original_video_height))
    
    # Initialize a deque to store frames for sequence input.
    frames_queue = deque(maxlen=SEQUENCE_LENGTH)
    
    # Initialize the predicted class name.
    predicted_class_name = ''
    
    # Process each frame in the video.
    while video_reader.isOpened():
        ok, frame = video_reader.read()
        if not ok:
            break
        
        # Resize the frame to the required input size.
        resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
        
        # Normalize the frame values.
        normalized_frame = resized_frame / 255
        
        # Add the normalized frame to the frames queue.
        frames_queue.append(normalized_frame)
        
        # If the queue is full, make a prediction.
        if len(frames_queue) == SEQUENCE_LENGTH:
            predicted_labels_probabilities = LRCN_model.predict(np.expand_dims(frames_queue, axis=0))[0]
            predicted_label = np.argmax(predicted_labels_probabilities)
            predicted_class_name = CLASSES_LIST[predicted_label]
        
        # Overlay the predicted class name on the frame.
        cv2.putText(frame, predicted_class_name, (10, 30), 
                    cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 255, 0), 2)
        
        # Write the frame to the output video file.
        video_writer.write(frame)
    
    # Release video readers and writers.
    video_reader.release()
    video_writer.release()
def predict_single_action(video_file_path, SEQUENCE_LENGTH):
    '''
    This function will predict a single action in a video using the LRCN model.
    Args:
    video_file_path:  The path of the video stored on disk for action recognition.
    SEQUENCE_LENGTH:  The fixed number of frames in a video passed as one sequence to the model.
    '''
    # Open the video file for reading.
    video_reader = cv2.VideoCapture(video_file_path)
    
    # Retrieve the original video dimensions.
    original_video_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_video_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Initialize an empty list to store frames.
    frames_list = []
    
    # Initialize the predicted class name.
    predicted_class_name = ''
    
    # Get the total number of frames in the video.
    video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate the skip window to evenly sample frames for the sequence.
    skip_frames_window = max(int(video_frames_count / SEQUENCE_LENGTH), 1)
    
    # Iterate through the frames to build the sequence.
    for frame_counter in range(SEQUENCE_LENGTH):
        # Set the video reader to the frame specified by the skip window.
        video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter * skip_frames_window)
        
        # Read the frame from the video.
        success, frame = video_reader.read()
        
        # Break if the frame reading was unsuccessful.
        if not success:
            break
        
        # Resize the frame to the desired input size.
        resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
        
        # Normalize the pixel values to the range [0, 1].
        normalized_frame = resized_frame / 255
        
        # Append the normalized frame to the list.
        frames_list.append(normalized_frame)
    
    # Perform action recognition on the sequence of frames.
    predicted_labels_probabilities = LRCN_model.predict(np.expand_dims(frames_list, axis=0))[0]
    
    # Determine the predicted action label.
    predicted_label = np.argmax(predicted_labels_probabilities)
    
    # Map the label to the corresponding action class name.
    predicted_class_name = CLASSES_LIST[predicted_label]
    
    # Display the predicted action and confidence.
    print(f'Action Predicted: {predicted_class_name}\nConfidence: {predicted_labels_probabilities[predicted_label]}')
    
    # Release the video reader.
    video_reader.release()
    
def main():
    st.image("activityrecog.jpeg")
    st.title("Action Recognition Streamlit App")

    # Input field for YouTube video URL
    youtube_url = st.sidebar.text_input("Enter YouTube Video URL:")
    button = st.sidebar.button("Recognize Human Activity")
    st.sidebar.write("This model can predict 6 activities: \n1. WalkingWithDog\n2. TaiChi\n3. Swing\n4. HorseRace\n5. Basketball\n6. PushUps")
    if button:
        # Download and predict on the YouTube video
        st.write("Downloading and processing the video. Please wait...")
        video_title = download_youtube_videos(youtube_url, "test_videos")
        input_video_file_path = f'test_videos/{video_title}.mp4'
        output_video_file_path = f'test_videos/{video_title}--Output-SeqLen{SEQUENCE_LENGTH}.mp4'
        predict_on_video(input_video_file_path, output_video_file_path, SEQUENCE_LENGTH)
        st.header('input video', divider='rainbow')
        st.video(input_video_file_path, format="video/mp4")
        st.header('output video', divider='rainbow')
        st.video(output_video_file_path, format="video/mp4")

if __name__ == "__main__":
    main()
