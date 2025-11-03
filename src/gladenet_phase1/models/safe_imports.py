# تحميل مباشر من ملفنا المحلي بدون أي مناورات مسارات
from .uformer_local import Uformer as UformerModel


# خيار: لو في أي كود قديم بينادي الدالة دي، نديها placeholder بسيط
def predict_img_with_smooth_windowing(*args, **kwargs):
    raise NotImplementedError("predict_img_with_smooth_windowing is not provided in this setup.")
