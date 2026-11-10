import vww_model

model = vww_model.mobilenet_v1()

model.load_weights("trained_models/vww_96.h5")

print("SUCCESS!")
print("Pretrained weights loaded.")