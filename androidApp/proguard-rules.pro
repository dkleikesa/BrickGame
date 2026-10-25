# ---------------- TensorFlow Lite / LiteRT ----------------
# 模型加载与推理全部经由 JNI，native 侧按类名反查 Java 对象；R8 改名会直接
# 让 Interpreter / Delegate 初始化失败（表现为一启动就 nil 或 UnsatisfiedLinkError）。
-keep class org.tensorflow.lite.** { *; }
-keep class com.google.ai.edge.litert.** { *; }
-dontwarn org.tensorflow.lite.**
-dontwarn com.google.ai.edge.litert.**

# GPU delegate 由 TFLite 运行时按字符串反射创建
-keep class org.tensorflow.lite.gpu.** { *; }
-keep class org.tensorflow.lite.nnapi.** { *; }
-dontwarn org.tensorflow.lite.gpu.**

# ---------------- 保留栈可读性 ----------------
# 线上崩溃日志要能定位到行，混淆后方法名不可读但行号仍有用
-keepattributes SourceFile,LineNumberTable
-renamesourcefileattribute SourceFile
