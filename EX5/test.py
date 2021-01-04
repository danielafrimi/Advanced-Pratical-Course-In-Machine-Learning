from EX5.models import GeneratorForMnistGLO

glo: GeneratorForMnistGLO = GeneratorForMnistGLO(code_dim=100)
glo.load('weights.ckpt')

# Check GLO on random input
glo.test()