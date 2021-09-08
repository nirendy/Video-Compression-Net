import streamlit as st


def select_input_text(key, label, options, on_change=None, use_text_area=False):
    if on_change is None:
        def nothing():
            pass

        on_change = nothing

    select_box_label = 'Existing ' + label
    if key not in st.session_state:
        st.session_state[key] = options[0]

    def set_predefined(val):
        st.session_state[key] = val
        on_change()

    text_input_container = st.container()

    with st.expander(select_box_label):
        for i, opt in enumerate(options):
            if i > 0:
                # st.write('___')
                pass

            if st.button(label=opt, key=f'{key}_button_choose_{i}'):
                set_predefined(opt)

            # col1, col2 = st.columns([9, 1])
            # col1.write(opt)
            # if col2.button(label='Choose', key=f'{key}_button_choose_{i}'):
            #     set_predefined(opt)

    with text_input_container:
        text_element = st.text_area if use_text_area else st.text_input
        text_element(
            label=label,
            key=key,
            on_change=on_change
        )
