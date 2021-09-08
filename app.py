import math
import time

import streamlit as st

from demo_app.components.text_with_predefined import select_input_text
from demo_app.consts.locale import Locale
import demo_app.utils.video_dataset as VideoDataset
from demo_app.stores import AppStateKeys
from demo_app.stores.app_state import AppState
from demo_app.utils.format_utils import humanbytes
from demo_app.utils.video_dataset import get_available_checkpoints_weights

ROWS_IN_PAGE = 5


def show_videos(col_name_video_path_tuples, image_speed):
    cols_count = len(col_name_video_path_tuples)
    if cols_count == 0:
        return
    col_names = [col_name_video_path_tuple[0] for col_name_video_path_tuple in col_name_video_path_tuples]
    video_paths = [col_name_video_path_tuple[1] for col_name_video_path_tuple in col_name_video_path_tuples]
    videos = [VideoDataset.get_frame_paths(video_path) for video_path in video_paths]
    frames_count = len(videos[0])
    page_num = 1
    with st.sidebar:
        if image_speed == 0:
            page_num = st.number_input(
                'Page Num',
                min_value=1,
                value=1,
                max_value=math.ceil(frames_count / ROWS_IN_PAGE),
                step=1
            ) - 1

    frame_container = st.empty()
    st.write('---')
    cols = st.columns(cols_count)
    for col_i, col_name in enumerate(col_names):
        with cols[col_i]:
            st.write(col_name)

    total_sizes = []
    image_containers = []
    for i, _ in enumerate(col_names):
        total_sizes.append(0)
        image_containers.append(
            None if image_speed == 0 else cols[i].empty()
        )

    for frame_i in range(frames_count):
        in_page = image_speed > 0 or (
                (frame_i >= page_num * ROWS_IN_PAGE)
                and
                (frame_i < (page_num + 1) * ROWS_IN_PAGE)
        )

        frame_display_num = frame_i + 1

        if in_page:
            if image_speed == 0:
                # pad_cols = st.container().columns(cols_count * 2 + 1)
                # frame_container = pad_cols[cols_count]
                st.write('---')
                frame_container = st.container()

            frame_container.write(f"Frame {frame_display_num}")

        if in_page:
            cols = st.columns(cols_count)

        for col_i, video in enumerate(videos):
            frame = video[frame_i]

            container = cols[col_i] if image_speed == 0 else image_containers[col_i].container()
            with container:
                size = frame.stat().st_size
                total_sizes[col_i] += size

                if in_page:
                    st.image(str(frame))
                    st.write(f"size = {humanbytes(size)} (Total: {humanbytes(total_sizes[col_i])})")

        if image_speed > 0:
            time.sleep(image_speed)


if __name__ == "__main__":
    st.set_page_config(
        page_title=Locale.app_title, page_icon=":memo:",
        layout='wide'
    )

    with st.sidebar:
        demo_name = st.text_input('Demo Name', value='try1')
        image_speed = st.number_input('Image Speed', min_value=0.0, value=0.0, step=0.01)

    if not demo_name:
        st.stop()
    demo = VideoDataset.Demo(demo_name)

    if not demo.input_path.exists():
        cols = st.columns([0.8, 0.2])
        with cols[0]:
            chosen_video = st.selectbox(
                "Choose Video",
                options=VideoDataset.get_videos_names(),
                format_func=lambda v: f"{v.parent.name}/{v.name}"
            )
        with cols[1]:
            if st.button('Load Video'):
                demo.load_input_video(chosen_video)
                st.experimental_rerun()

        show_videos([('', chosen_video)], image_speed)
        st.stop()

    checkbox_order = [
        ('Original Video', demo.input_path, True, ''),
        ('Reconstructed Video', demo.reconstructed_path, False),
        ('Reconstructed Diff', demo.reconstructed_diff_path, False),
    ]
    with st.sidebar:
        st.write('---')

        weight_file = None
        if demo.fine_tune_path.exists():
            fine_tuned_weight_file = demo.get_finetune_weights()
            st.write(f"Weight File is = {fine_tuned_weight_file.name[10 + 15:][:-4]}")
        else:
            weight_file = st.selectbox(
                label="Choose Weight file",
                help="Higher lambda => lower distortion, higher bitrate",
                options=get_available_checkpoints_weights(),
                index=10,
                format_func=lambda x: x.name[15:][:-4]
            )

        frequency = None
        if demo.compression_path.exists():
            st.write(f"Frequency is = {demo.get_frequency()}")
        else:
            frequency = st.number_input(
                label="Frequency",
                min_value=1,
                value=7,
                step=1
            )

        show_col = [
            st.checkbox(
                label=(
                        f'Show {checkbox_details[0]}' +
                        ('' if checkbox_details[1].exists() else f' (Not Calculated)')
                ),
                value=checkbox_details[2]
            )
            for checkbox_details in checkbox_order
        ]
    chosen_columns = [checkbox_order[i] for i, v in enumerate(show_col) if v]

    for i, chosen_column in enumerate(chosen_columns):
        need_rerun = False
        if not chosen_column[1].exists():
            demo.run(
                weight_file=weight_file,
                frequency=None if frequency is None else int(frequency)
            )
            st.experimental_rerun()

    show_videos(chosen_columns, image_speed)
