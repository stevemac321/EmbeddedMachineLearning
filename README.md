# Embedded Machine Learning Project - STM32 X-Cube-AI Integration

## Overview

This project demonstrates the integration of an **STM32 X-Cube-AI** project with the **TensorFlow Text Classification** model. It uses the `text_classification_dense_input.h5` model and `run_inference.py` script to verify the parity of inference results between the STM32 environment and a Python environment.

### References

1. **TensorFlow Text Classification Tutorial**: The model used in this project is based on the text classification tutorial provided by TensorFlow. You can find more details and explore the tutorial [here](https://www.tensorflow.org/tutorials/keras/text_classification).
2. **X-Cube-AI Data Brief**: For a summary of the X-Cube-AI expansion package and its capabilities, refer to the [X-Cube-AI Data Brief](https://www.st.com/resource/en/data_brief/x-cube-ai.pdf).
3. **Getting Started with X-Cube-AI**: For a more comprehensive guide on using the X-Cube-AI expansion package, refer to the [User Manual](https://www.st.com/resource/en/user_manual/dm00570145-getting-started-with-xcubeai-expansion-package-for-artificial-intelligence-ai-stmicroelectronics.pdf).
4. **ARM ML Zoo**: For a variety of machine learning models, including the one used in this project, visit the [ARM ML Zoo](https://github.com/ARM-software/ML-zoo).
5. **STM32 AI Model Zoo**: For additional STM32 models and reference examples, visit the [STM32 AI Model Zoo](https://github.com/STMicroelectronics/stm32ai-modelzoo).

## Project Details

### Hardware Requirements

- **Board**: NUCLEO-H723ZG
  - Note: The model used in this project requires significant memory and processing power, so using an STM32F4xx board may not be sufficient due to the model size, RAM requirements, and processor speed.
  
### Software and Tools

- **Development Environment**: STM32CubeIDE with X-Cube-AI plugin.
- **Operating System**: Windows (for this demo).
- **Additional Software**:
  - [Graphviz](https://graphviz.gitlab.io/download/) for generating diagrams with Doxygen.
  - [Terminal](https://sites.google.com/site/terminalbpp/) or a similar serial port reader for viewing output through the ST-Link Virtual Port.

### Model Modification and Setup

The original model used in the TensorFlow text classification tutorial included an **embedding layer**, which was not compatible with the STM32 X-Cube-AI framework. To resolve this, the model was modified to use a **dense-only architecture**. The resulting `text_classification_dense_input.h5` model was then imported into STM32CubeIDE using the X-Cube-AI plugin.

The modification involved mapping the text inputs directly to float values (`0.8`, `0.5`, `0.2`) using the Python script `run_inference.py` to ensure consistent input features. This approach allowed us to achieve the same inference results as the original Python model without needing complex embedding or tokenization processes on the STM32 platform.

### Project Structure

- **Root Directory**:
  - `text_classification_dense_input.h5`: The Keras model file used for the project.
  - `run_inference.py`: The Python script used to map the `examples[]` array of text inputs to corresponding float values and verify parity between Python and STM32 results.
  - `Doxyfile`: The Doxygen configuration file for generating project documentation.
  
- **`App` Directory**:
  - Contains the main X-Cube-AI application source files, including the `app_x-cube-ai.c` file where the primary inference functions `acquire_and_process_data()` and `post_process()` are implemented.

- **`output_dir` Directory**:
  - Contains the `network_generate_report` and other output files generated by X-Cube-AI, which provide information about the model structure, memory usage, and more. These files are included for convenience and Doxygen generation.

### Functionality

The main functionality of this project is to achieve **parity between Python and STM32 inference results**. The inference code resides in the `USER CODE` sections of `app_x-cube-ai.c` under the following functions:

1. **`acquire_and_process_data(ai_i8* data[])`**:
   - This function maps predefined float values to the input buffer of the model, based on the text descriptions:
     - `"The movie was great!"` → `0.8`
     - `"The movie was okay."` → `0.5`
     - `"The movie was terrible..."` → `0.2`

2. **`post_process(ai_i8* data[])`**:
   - This function reads the model's predictions and outputs the results through UART. The results can be viewed using a serial port reader such as **Terminal**.

### Viewing Output

- The program outputs results via **USART3** using default baud rates. The serial output can be read using a serial port reader like **Terminal**. Ensure that the ST-Link Virtual Port settings in the Windows Device Manager and the Terminal baud rates match the defaults used in the STM32 code.

### Documentation

- **Doxygen Documentation**:
  - A `Doxyfile` is provided at the root of the project. To generate the documentation:
    1. Open the Doxyfile in any text editor and set `OUTPUT_DIRECTORY` to your preferred location.
    2. Ensure `DOT_PATH` is set to the Graphviz installation path on your system (e.g., `C:/Program Files (x86)/Graphviz/bin`).
    3. Run the following command in the terminal:
       ```bash
       doxygen Doxyfile
       ```
  - This will generate HTML and other documentation formats based on the project files (`*.h`, `*.c`, `*.txt`, `*.json`, `*.csv`) located in the `App` directory.

### Summary

This project successfully demonstrates the use of X-Cube-AI to deploy a text classification model on the STM32 NUCLEO-H723ZG board. It achieves parity with Python-based inference using the provided `run_inference.py` script. The results of the inference are printed via UART and can be viewed using a serial port reader.

By following the instructions and references provided, users can replicate the project and modify it for their own applications.

