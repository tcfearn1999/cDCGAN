# Fashion MNIST Conditional Generative Adversarial Network (CGAN)

This Python script implements a Conditional Generative Adversarial Network (CGAN) using the Fashion MNIST dataset. CGAN is a type of generative model that can generate images conditioned on a given 
label.

## Requirements:

- Python 3.x
- numpy 1.21.5
- keras 2.6.1

Install required libraries using pip:

```bash
pip install numpy==1.21.5 keras==2.6.1
```

## Instructions:

1. Download the script and fashion_mnist_cgan.py file.

2. Run the script using Python:

```bash
python fashion_mnist_cgan.py
```

3. The script will train the CGAN model using the Fashion MNIST dataset. During training, it will display the loss values for the discriminator and the generator.

4. Once training is complete, the trained generator model will be saved as 'cgan_generator.keras' file.

5.After training, the script will be able to generate output images. Here's an example of the generated output:

![Generated after training for 300 epochs](/ouput/Ouput_300.png)
