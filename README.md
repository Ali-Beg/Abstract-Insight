# Abstract-Insight

```markdown
# Medical Abstract Classification App 🏥🧠

This **Medical Abstract Classification** Streamlit app allows users to classify sentences in medical abstracts into categories such as **Objective**, **Methods**, **Results**, **Conclusions**, and **Background**. The app utilizes a machine learning model to predict the labels for each sentence in the abstract based on its content.


## 📝 Features

- Classify medical abstracts sentence by sentence into distinct categories.
- Displays both the predicted **label** and **probability** for each sentence.
- Organized output with separate sections for each label.
- Simple, user-friendly interface built with **Streamlit**.

## 🚀 Demo

Try the live version of the app here: [Abstract-Insight](https://abstract-insight.streamlit.app/) 

## 📂 Project Structure

```
├── app.py                   # Main Streamlit app script
├── model/                   # Directory containing the trained model
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
```

## 📦 Installation & Setup

To run this project locally, follow these steps:

1. **Clone the repository**:

   ```bash
   git clone https://github.com/your-username/my-streamlit-app.git
   cd my-streamlit-app
   ```

2. **Create a virtual environment** and activate it:

   ```bash
   python -m venv env
   source env/bin/activate      # On Linux/MacOS
   .\env\Scripts\activate       # On Windows
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Streamlit app**:

   ```bash
   streamlit run app.py
   ```

## ⚙️ Usage

- Open the app in your browser.
- Input a medical abstract in the text box.
- Press the "Classify Abstract" button.
- The app will display the classification of each sentence into the categories: **Objective**, **Methods**, **Results**, **Conclusions**, or **Background** with the corresponding probability.

## 🛠 Dependencies

The following packages are required to run the project:

- `streamlit`
- `tensorflow`
- `pandas`
- `numpy`

The full list of dependencies can be found in `requirements.txt`.

## 🤖 Model Information

The machine learning model used in this project was trained on a dataset of medical abstracts to classify sentences into five categories:
- **Objective**
- **Methods**
- **Results**
- **Conclusions**
- **Background**

The model is loaded from a saved `.keras` file in the project directory.

## 🤝 Contributing

Contributions are welcome! Feel free to fork this repository and submit pull requests with improvements.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

For any questions or issues, feel free to reach out:

- Email: mbeg937@gmail.com
- GitHub: [ALI BEG](https://github.com/Ali-Beg)


