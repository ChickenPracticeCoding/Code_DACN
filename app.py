import streamlit as st
import os
import cv2
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
from PIL import Image

# Load mô hình đã train từ file .h5
model = load_model('E:\\Hoctap\\DACN\\DACN_Thamkhao\\model_lenet.h5')


# Danh sách các nhãn bệnh ngoài da
classes = {
    0: ('akiec', 'Actinic keratoses and intraepithelial carcinomae'),
    1: ('bcc', 'Basal cell carcinoma'),
    2: ('bkl', 'Benign keratosis-like lesions'),
    3: ('df', 'Dermatofibroma'),
    4: ('nv', 'Melanocytic nevi'),
    5: ('vasc', 'Pyogenic granulomas and hemorrhage'),
    6: ('mel', 'Melanoma')
}


# Hàm dự đoán và hiển thị thông tin chi tiết
def predict_and_display(image, model):
    # Chuyển đổi PIL image sang định dạng cv2
    img = np.array(image)

    # Resize ảnh về kích thước mà mô hình yêu cầu (ví dụ: 28x28)
    img_resized = cv2.resize(img, (224, 224))

    # Chuẩn hóa dữ liệu ảnh
    img_normalized = img_resized / 255.0

    # Dự đoán
    result = model.predict(img_normalized.reshape(1, 224, 224, 3))
    max_prob = max(result[0])
    class_ind = list(result[0]).index(max_prob)
    class_name = classes[class_ind]

    return class_name, max_prob, result[0]


# Giao diện người dùng
def main():
    st.set_page_config(page_title="Dự Đoán Bệnh Da", page_icon=":microscope:", layout="wide")

    # Tiêu đề chính
    st.markdown("<h1 style='text-align: center; color: #007bff;'>Dự Đoán Bệnh Da</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align: center; color: #6c757d;'>Tải lên một hình ảnh về tình trạng da, mô hình AI của chúng tôi sẽ dự đoán bệnh với các xác suất chi tiết.</p>",
        unsafe_allow_html=True)

    # Tạo cột để cân đối giao diện
    col1, col2, col3 = st.columns([1, 6, 1])

    with col2:
        uploaded_file = st.file_uploader("Chọn một hình ảnh...", type=["jpg", "jpeg", "png"],
                                         label_visibility="visible")
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption='Hình ảnh đã tải lên.', use_column_width=True)

            # Nút để dự đoán
            if st.button('Dự Đoán', key='predict_button'):
                class_name, max_prob, probabilities = predict_and_display(image, model)

                # Hiển thị kết quả dự đoán
                st.markdown(f"<h2 style='text-align: center; color: #FF6347;'>Bệnh Dự Đoán: {class_name[1]}</h2>",
                            unsafe_allow_html=True)
                st.markdown(f"<h4 style='text-align: center;'>Xác Suất Cao Nhất: {max_prob:.4f}</h4>",
                            unsafe_allow_html=True)

                st.markdown("<h4 style='text-align: center;'>Xác Suất Theo Các Loại Bệnh:</h4>", unsafe_allow_html=True)
                for i, (key, value) in enumerate(classes.items()):
                    st.markdown(f"<p style='text-align: center;'>{value[1]}: {probabilities[i]:.4f}</p>",
                                unsafe_allow_html=True)

                # Hiển thị ảnh kèm nhãn dự đoán
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.imshow(cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB))
                ax.set_title(f'Loại Bệnh Dự Đoán: {class_name[1]}', fontsize=16, color='#FF6347')
                ax.axis('off')
                st.pyplot(fig)


if __name__ == '__main__':
    main()
