import streamlit as st
from guided_search import GuidedSearchController, Rating


LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# --------------- #
# Utils functions #
# --------------- #

# Keep track of the current screen.
def reset():
    st.session_state.screen = 0
    st.session_state.subscreen = 0
    st.session_state.initialized = True
    st.session_state.gs = GuidedSearchController()

# Define a function that will be called to process the next screen.
def next_screen():
    st.session_state.screen += 1
    st.session_state.subscreen = 0
    if st.session_state.screen > 2:
        st.session_state.screen = 1

# Update the configs of the guided search.
def update_bwd_configs():
    st.session_state.gs.update_bwd_configs(
        st.session_state.num_candidates,
        st.session_state.bwd_temperature,
        st.session_state.bwd_max_tokens,
    )

# Update the configs of the outputs generation.
def update_fwd_configs():
    st.session_state.gs.update_fwd_configs(
        st.session_state.fwd_model,
        st.session_state.fwd_temperature,
        st.session_state.fwd_max_tokens,
    )

# Initialize the GuidedSearch
def setup_guided_search():
    st.session_state.gs.setup(
        st.session_state.meta_prompt,
        st.session_state.input_examples
    )
    next_screen()

def provide_feedback():
    st.session_state.gs.feedback = st.session_state.feedback
    next_screen()
    st.session_state.gs.prompt_proposal_step()
    st.session_state.gs.inference_candidates_per_example()

def submit_ratings():
    selected_outputs = {
        (j, i): (
            getattr(st.session_state, f"score_{i}_{j}"),
            getattr(st.session_state, f"feedback_{i}_{j}", None)
        )
        for i, input in enumerate(st.session_state.gs.examples)
        for j, prompt in enumerate(st.session_state.gs.prompt_candidates_outputs)
    }
    st.session_state.gs.consolidate_prompts(selected_outputs)
    next_screen()

if 'initialized' not in st.session_state:
    reset()


# ------------------ #
# Main Streamlit app #
# ------------------ #
st.header('GUIDE: A tool for guided meta-prompt search')

############
# Sidebar  #
############

with st.sidebar:
    with st.form(key="bwd-configs"):
        st.markdown("**GUIDE Settings**", help="Good luck")
        st.slider("Number of meta-prompt candidates", 1, 5, st.session_state.gs.num_candidates, key="num_candidates")
        st.slider("Search diversity", 0.0, 1.0, st.session_state.gs.bwd_temperature, key="bwd_temperature",
                  help="Intuitively, lower 'Search diversity' values will lead to more deterministic and less creative meta-prompts.")
        st.slider("Max meta-prompt length", 0, 500, st.session_state.gs.bwd_max_tokens, key="bwd_max_tokens", help="aka max tokens")
        st.toggle("Show meta-prompts", key="show_metaprompts", help="Show the meta-prompts used to generate the outputs.")
        st.form_submit_button(
            label="Apply",
            on_click=update_bwd_configs,
            type="primary",
            use_container_width=True,
        )
    with st.form(key="fwd-configs"):
        st.markdown("#### Output Generation Settings", help="Settings used in your application")
        st.selectbox("Model", st.session_state.gs.available_models, key="fwd_model", help="LLM used in your application to generate output.")
        st.slider("Temperature", 0.0, 1.0, st.session_state.gs.fwd_temperature, key="fwd_temperature", help="Temperature used for the generated output.")
        st.slider("Max tokens", 0, 500, st.session_state.gs.fwd_max_tokens, key="fwd_max_tokens", help="Max number of tokens for generated output.")
        st.form_submit_button(
            label="Apply",
            on_click=update_fwd_configs,
            type="primary",
            use_container_width=True,
        )
    st.markdown(f"Search iteration {st.session_state.gs.optimization_step}")
    with st.expander("#### Meta-Prompt History", expanded=True):
        for idx, (prompt, example_outputs) in enumerate(st.session_state.gs.history.items(), 1):
            st.markdown(f"##### Meta-Prompt {idx}: {prompt}")
            for example_output in example_outputs:
                st.markdown(f"<div style='color: #666; font-size: 12px; margin-bottom: 5px;'><b>{example_output.example} | {example_output.output}", unsafe_allow_html=True)
            st.markdown("<div style='margin-bottom: 10px;'", unsafe_allow_html=True)

############
# Screen 0 #
############
if st.session_state.screen == -1:
    with st.form(key="welcome_form"):

        submit_button = st.form_submit_button(
            label="Start GUIDING",
            on_click=next_screen,
            use_container_width=True,
            type="primary",
        )

if st.session_state.screen == 0:
    st.markdown("GUIDE is a prototype tool to search for alternative meta-prompts. To do so please put your current meta-prompt in the top textbox, and a few example input prompts that you want to test in the bottom textbox.")
    with st.form(key="init_form"):
        # Text area for initial meta-prompt.
        st.markdown("##### Enter an initial meta-prompt to start from")
        meta_prompt = st.text_area(
            'Enter a meta-prompt',
            key="meta_prompt",
            value="",
            label_visibility="collapsed",
            placeholder="",
        )

        # Text area for a list of input examples (i.e., expected user queries). Each example is separated by a new line.
        st.markdown("##### Enter a few input examples (one per line)")
        input_examples = st.text_area(
            'Enter a few input examples (one per line)',
            key="input_examples",
            value="",
            label_visibility="collapsed",
            placeholder="",
        )
        # Split the input examples by new line.
        input_examples = input_examples.split('\n')
        submit_button = st.form_submit_button(
            label="Start guided search",
            on_click=setup_guided_search,
            use_container_width=True,
            type="primary",
        )

############
# Screen 1 #
############
if st.session_state.screen == 1:
    st.subheader('Provide feedback to guide the meta-prompt search')
    if st.session_state.gs.optimization_step > 1:
        # After the first iteration of optimization, let the user know we are starting a new round of search/optimization.
        st.info("Here is the new meta-prompt based on your preference ratings. You can check out the outputs and choose to do another round of search.")

    with st.form(key="feedback_form"):
        # Display the meta-prompt used in a code format.
        st.markdown(f"\n| Current meta-prompt |\n|-|\n|{st.session_state.gs.meta_prompt}|")
        st.markdown("<style> table { width: 100%; } </style>", unsafe_allow_html=True)

        # Display the results of the inference formatted in a markdown table. Each row is an input-output pair.
        # The first column is the input, and the second column is the output.
        st.write("")
        st.markdown("""\
Input | Output
--- | ---
""" + "\n".join(f'{out.example} | {out.output}' for out in st.session_state.gs.example_outputs))

        # Add newlines
        st.markdown('')
        st.markdown('')
        st.write('Based on the results, provide feedback to guide the search for new meta-prompts.')
        # Add text area for feedback on the metaprompt.
        feedback = st.text_area(
            label="Based on the results, provide feedback to guide the meta-prompt search:",
            key="feedback",
            label_visibility="collapsed",
            value=st.session_state.gs.feedback or "",
            placeholder="",
        )

        submit_button = st.form_submit_button(
            label="Generate prompt candidates",
            on_click=provide_feedback,
            use_container_width=True,
            type="primary",
        )


############
# Screen 2 #
############
if st.session_state.screen == 2:
    st.subheader('Rate outputs generated by different meta-prompt candidates')
    st.markdown(
        'Guide the Meta-Prompt search by providing feedback on '
        'the outputs generated with different meta-prompt candidates. '
        'You are not required to provide feedback on all example outputs. '
        'The outputs skipped will not inform the generation of the new meta prompt.'
    )

    st.write("---")

    # Go over each input example and generate the output from inference_per_example function in guided_search.py.
    # Then, display the input, output, and a checkbox for the user to indicate whether they prefer the output.
    st.session_state.selected_output = []
    for i, input in enumerate(st.session_state.gs.examples):
        # Display the input as label.
        st.markdown(f":red[**Input {i + 1}:** {input}]")

        outputs = st.session_state.gs.find_outputs_by_example(input)
        for j, output in enumerate(outputs):
            st.caption(f"Meta-prompt {LETTERS[j]}\: " + list(st.session_state.gs.prompt_candidates_outputs)[j] if st.session_state.get("show_metaprompts") else "")
            st.markdown(f"**Output {LETTERS[j]}:** {output}")

            radio = st.radio(
                "Rating",
                Rating.values(),
                format_func=lambda e: Rating.icons(e),
                horizontal=True,
                index=2,
                key=f"score_{i}_{j}",
                label_visibility="collapsed"
            )
            if radio in (1, -1):
                st.text_input(
                    label="Provide feedback on the output:",
                    key=f"feedback_{i}_{j}",
                    label_visibility="collapsed",
                    value="",
                    placeholder="Provide feedback on the output (optional)",
                )

        st.write("---")

    submit_button = st.button(
        label="Submit my preferences",
        on_click=submit_ratings,
        use_container_width=True,
        type="primary",
    )
