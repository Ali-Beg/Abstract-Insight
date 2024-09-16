# 🏥 Medical Abstract Classification App

This **Medical Abstract Classification** app allows users to classify sentences in medical abstracts into categories like **Objective**, **Methods**, **Results**, **Conclusions**, and **Background** using a machine learning model. The app is built with **Streamlit**

## 🌟 Features

- Classifies medical abstracts into five distinct categories.
- Displays the predicted label and probability for each sentence.
- Clean and simple UI with Streamlit.

## 🚀 Demo

Try the live app: [Abstract_Insight](https://abstract-insight.streamlit.app/)

## 📂 Project Structure

```bash
my-streamlit-app/
├── app.py                 # Main Streamlit app script
├── model/                 # Trained model directory
├── requirements.txt       # Dependencies for the project
└── README.md              # Documentation (this file)
```

## 🛠️ Setup Instructions

To run this project locally:

1. **Clone the repository**:

    ```bash
    git clone https://github.com/Ali-Beg/Abstract-Insight.git
    cd Abstract-Insight
    ```

2. **Create and activate a virtual environment**:

    On Windows:
    ```bash
    python -m venv env
    .\env\Scripts\activate
    ```

    On Linux/Mac:
    ```bash
    python3 -m venv env
    source env/bin/activate
    ```

3. **Install the required dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

4. **Run the Streamlit app**:

    ```bash
    streamlit run app.py
    ```

## 🧠 Model Information

The model is trained to classify medical abstract sentences into one of five categories:
- **Objective**
- **Methods**
- **Results**
- **Conclusions**
- **Background**

The model is loaded from a `.keras` file located in the `model` directory.

## 📋 Requirements

- `streamlit`
- `tensorflow`
- `pandas`
- `numpy`

Make sure all dependencies are listed in `requirements.txt`.

## 🤝 Contributing

Feel free to fork the repository and submit pull requests! Contributions are welcome.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ✉️ Contact

For any questions or support, contact:
- GitHub: [Ali Beg](https://github.com/Ali-Beg)
- Email: mbeg937@gmail.com
