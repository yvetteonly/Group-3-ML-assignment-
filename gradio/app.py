import gradio as gr
import pickle

def predict_price(house_size, num_rooms):
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    prediction = float(model.predict([[house_size, num_rooms]])[0])
    return f"Predicted Price: {prediction:.2f}"

with gr.Blocks(css="body { display: flex; justify-content: center; align-items: center; height: 100vh; }") as demo:
    gr.Markdown("# 🏡 House Price Prediction")
    gr.Markdown("Enter the house size and number of rooms to estimate the price.")
    with gr.Row():
        house_size = gr.Number(label="House Size (sq ft)", interactive=True)
        num_rooms = gr.Number(label="Number of Rooms", interactive=True)
    predict_btn = gr.Button("Predict")
    prediction_output = gr.Textbox(label="Prediction (RWF)")
    predict_btn.click(predict_price, inputs=[house_size, num_rooms], outputs=prediction_output)

demo.launch(share=True)
