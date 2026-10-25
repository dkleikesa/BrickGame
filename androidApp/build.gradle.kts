import java.io.FileInputStream
import java.util.Properties
import org.jetbrains.kotlin.gradle.dsl.JvmTarget

// 发签密码放 androidApp/keystore.properties（已 gitignore），不入库
val keystorePropsFile = rootProject.file("androidApp/keystore.properties")
val keystoreProps = Properties().apply {
    if (keystorePropsFile.exists()) FileInputStream(keystorePropsFile).use { load(it) }
}

plugins {
    alias(libs.plugins.androidApplication)
    alias(libs.plugins.composeCompiler)
}

kotlin {
    compilerOptions {
        jvmTarget = JvmTarget.JVM_11
    }
}
dependencies {
    implementation(project(":shared"))

    implementation(libs.androidx.activity.compose)
    // 分享留档 ZIP 要 FileProvider（androidx.core.content）
    implementation(libs.androidx.core.ktx)
    // 物品识别模型（brick_embedder.tflite）的运行时，模型与原型在 src/main/assets/
    implementation(libs.tensorflow.lite)
    // 识别跑在 GPU delegate 上（三者里最快），见 ml/TfliteTileClassifier
    implementation(libs.tensorflow.lite.gpu)
    implementation(libs.tensorflow.lite.gpu.api)

    implementation(libs.compose.runtime)
    implementation(libs.compose.foundation)
    implementation(libs.compose.material3)
    implementation(libs.compose.ui)
    implementation(libs.compose.uiToolingPreview)
    debugImplementation(libs.compose.uiTooling)
}

android {
    namespace = "com.qcmian.brickgame"
    compileSdk = libs.versions.android.compileSdk.get().toInt()

    defaultConfig {
        applicationId = "com.qcmian.brickgame"
        minSdk = libs.versions.android.minSdk.get().toInt()
        targetSdk = libs.versions.android.targetSdk.get().toInt()
        versionCode = 4
        versionName = "1.0.3"
        ndk {
            //noinspection ChromeOsAbiSupport
            abiFilters += listOf("arm64-v8a")
        }
    }
    packaging {
        resources {
            excludes += "/META-INF/{AL2.0,LGPL2.1}"
        }
    }
    signingConfigs {
        if (keystorePropsFile.exists()) {
            create("release") {
                storeFile = rootProject.file("androidApp/" + keystoreProps.getProperty("storeFile"))
                storePassword = keystoreProps.getProperty("storePassword")
                keyAlias = keystoreProps.getProperty("keyAlias")
                keyPassword = keystoreProps.getProperty("keyPassword")
                // 两种签名方案都出，新机型走 v2/v3，老机型回退 v1
                enableV1Signing = true
                enableV2Signing = true
            }
        }
    }
    buildTypes {
        release {
            signingConfig = signingConfigs.findByName("release")
            isMinifyEnabled = true
            isShrinkResources = true
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }
    buildFeatures {
        compose = true
    }
}