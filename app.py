import os
# Disable GPU completely before importing TensorFlow
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logs

import streamlit as st

# Set page config first
st.set_page_config(
    page_title="MNIST Adversarial Attack",
    page_icon="🛡️",
    layout="centered"
)

st.title("Effect of different Epsilon on accuracy of MNIST Dataset")

# Cache heavy imports
@st.cache_resource
def load_libraries():
    import tensorflow as tf
    import numpy as np
    from keras.datasets import mnist
    from keras.models import Sequential, Model
    from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
    import matplotlib.pyplot as plt
    import pandas as pd
    import warnings
    from art.attacks.evasion import FastGradientMethod
    from art.estimators.classification import KerasClassifier
    
    warnings.filterwarnings('ignore')
    return tf, np, mnist, plt, FastGradientMethod, KerasClassifier

# Load libraries once
tf, np, mnist, plt, FastGradientMethod, KerasClassifier = load_libraries()

# Cache data loading
@st.cache_data
def load_test_data():
    """Load and preprocess only test data to reduce memory usage"""
    try:
        # Load only test data
        (_, _), (test_images, test_labels) = mnist.load_data()
        
        # Use only first 1000 samples for faster processing
        test_images = test_images[:1000]
        test_labels = test_labels[:1000]
        
        # Preprocess
        test_images = test_images.reshape(-1, 28, 28, 1).astype('float32') / 255.0
        
        return test_images, test_labels
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None

# Cache model loading
@st.cache_resource
def load_model():
    """Load the pre-trained model"""
    try:
        model = tf.keras.models.load_model("mnist_model.h5")
        # Create ART KerasClassifier
        classifier = KerasClassifier(model=model, clip_values=(0, 1))
        return model, classifier
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.info("Make sure 'mnist_model.h5' file is in your repository")
        return None, None

# Load data and model
with st.spinner("Loading data and model..."):
    test_images, test_labels = load_test_data()
    model, classifier = load_model()

if test_images is None or model is None:
    st.stop()

class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

@st.cache_data
def generate_adversarial_examples(epsilon, _test_images, _test_labels, _classifier, _model):
    """Generate adversarial examples with caching"""
    try:
        # Generate adversarial examples
        attack = FastGradientMethod(estimator=_classifier, eps=epsilon)
        x_test_adv = attack.generate(x=_test_images)

        # Evaluate on clean and adversarial examples
        loss_clean, accuracy_clean = _model.evaluate(_test_images, _test_labels, verbose=0)
        loss_adv, accuracy_adv = _model.evaluate(x_test_adv, _test_labels, verbose=0)

        return accuracy_clean, accuracy_adv, x_test_adv
    except Exception as e:
        st.error(f"Error generating adversarial examples: {str(e)}")
        return None, None, None

# User interface
st.markdown("### Adversarial Attack Parameters")
epsilon = st.slider(
    "Epsilon value for FGSM Attack", 
    min_value=0.0, 
    max_value=1.0,  # Reduced max value 
    value=0.1,
    step=0.01,
    help="Higher epsilon = stronger attack but more visible perturbations"
)

if st.button("Generate Adversarial Examples", type="primary"):
    with st.spinner(f"Generating adversarial examples with ε={epsilon}..."):
        acc_clean, acc_adv, test_adv = generate_adversarial_examples(
            epsilon, test_images, test_labels, classifier, model
        )
        
        if acc_clean is not None:
            # Display results
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Epsilon (ε)", f"{epsilon:.2f}")
            
            with col2:
                st.metric("Clean Accuracy", f"{acc_clean:.3f}")
            
            with col3:
                st.metric("Adversarial Accuracy", f"{acc_adv:.3f}", 
                         delta=f"{acc_adv - acc_clean:.3f}")
            
            # Show attack success rate
            attack_success_rate = (acc_clean - acc_adv) / acc_clean * 100
            st.metric("Attack Success Rate", f"{attack_success_rate:.1f}%")
            
            # Visual comparison
            st.markdown("### Visual Comparison")
            
            with st.spinner("Generating predictions and visualizations..."):
                # Predictions
                pred_clean = np.argmax(model.predict(test_images[:10], verbose=0), axis=1)
                pred_adv = np.argmax(model.predict(test_adv[:10], verbose=0), axis=1)
                
                # Create visualization
                fig, axes = plt.subplots(2, 10, figsize=(15, 4))
                
                for i in range(10):
                    # Clean image
                    axes[0, i].imshow(test_images[i].reshape(28, 28), cmap="gray")
                    axes[0, i].set_title(
                        f"P:{pred_clean[i]}\nT:{test_labels[i]}",
                        color=("green" if pred_clean[i] == test_labels[i] else "red"),
                        fontsize=8
                    )
                    axes[0, i].axis("off")

                    # Adversarial image
                    axes[1, i].imshow(test_adv[i].reshape(28, 28), cmap="gray")
                    axes[1, i].set_title(
                        f"P:{pred_adv[i]}\nT:{test_labels[i]}",
                        color=("green" if pred_adv[i] == test_labels[i] else "blue"),
                        fontsize=8
                    )
                    axes[1, i].axis("off")

                axes[0, 0].set_ylabel("Clean", fontsize=10, rotation=0, ha='right')
                axes[1, 0].set_ylabel("Adversarial", fontsize=10, rotation=0, ha='right')
                fig.suptitle(f"Clean vs Adversarial Images (ε={epsilon})", fontsize=14)
                plt.tight_layout()
                
                st.pyplot(fig, use_container_width=True)
                
                # Statistics
                correct_clean = np.sum(pred_clean == test_labels[:10])
                correct_adv = np.sum(pred_adv == test_labels[:10])
                
                st.markdown("### Sample Statistics (First 10 images)")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"✅ **Correct (Clean):** {correct_clean}/10")
                    st.write(f"❌ **Incorrect (Clean):** {10 - correct_clean}/10")
                
                with col2:
                    st.write(f"✅ **Correct (Adversarial):** {correct_adv}/10")
                    st.write(f"❌ **Incorrect (Adversarial):** {10 - correct_adv}/10")

# Information sidebar
with st.sidebar:
    st.markdown("### About")
    st.markdown("""
    This app demonstrates the **Fast Gradient Sign Method (FGSM)** 
    adversarial attack on MNIST handwritten digit classification.
    
    **Key Points:**
    - Higher ε = stronger attack
    - Stronger attacks = lower adversarial accuracy
    - Visual changes may be subtle but effective
    """)
    
    st.markdown("### Model Info")
    if model is not None:
        st.write(f"Model loaded successfully")
        st.write(f"Test samples: {len(test_images) if test_images is not None else 0}")
    
    st.markdown("### Legend")
    st.markdown("""
    - **P:** Prediction
    - **T:** True label  
    - **Green:** Correct prediction
    - **Red/Blue:** Incorrect prediction
    """)
