package com.example.crossmediaapp.utils

import android.content.Context
import android.graphics.Bitmap
import android.hardware.camera2.params.RggbChannelVector
import android.os.Build
import android.util.Log
import androidx.annotation.RequiresApi
import org.opencv.android.Utils
import org.opencv.core.Core.*
import org.opencv.core.CvType
import org.opencv.core.CvType.*
import org.opencv.core.Mat
import org.opencv.core.Rect
import org.opencv.utils.Converters.Mat_to_vector_double
import org.opencv.imgcodecs.Imgcodecs
import org.opencv.imgproc.Imgproc
import org.opencv.imgproc.Imgproc.*
import java.io.File
import java.nio.ByteBuffer
import java.util.*
import kotlin.math.pow

/**
 * Demosaics the bayer-filter. The input is a 16-bit single-channel image, and the output
 * is a three-channel image.
 */
@RequiresApi(Build.VERSION_CODES.O)
fun demosaic(height: Int, width: Int, data: ByteBuffer): Mat {
    val img: Mat = Mat(height, width, CV_16UC1, data)
    var outputMat: Mat = Mat()
    Imgproc.demosaicing(img, outputMat, COLOR_BayerBG2BGR)

    balanceColors(outputMat).copyTo(outputMat)
    convertFrom16bitTo8bit(outputMat).copyTo(outputMat)
    var tempMat = Mat()
    outputMat.copyTo(tempMat)
    convertToXYZ(outputMat).copyTo(outputMat)
    convertToLab(outputMat).copyTo(outputMat)

    return outputMat
}



/**
 * Weighs the colours semi-appropriately to try and balance the colours in the image.
 */
fun balanceColors(matToBalance: Mat): Mat {
    val rScalar = 0.75
    val gScalar = 0.65
    val bScalar = 1.30

    var multiMat = Mat(matToBalance.rows(), matToBalance.cols(), CV_16UC3)
    var outputMat = Mat()
    matToBalance.copyTo(multiMat)
    var channels = mutableListOf<Mat>(Mat())

    split(multiMat, channels)
    channels[0] = channels[0].mul(Mat.ones(channels[0].rows(), channels[0].cols(), CV_16UC1), rScalar)
    channels[1] = channels[1].mul(Mat.ones(channels[1].rows(), channels[1].cols(), CV_16UC1), gScalar)
    channels[2] = channels[2].mul(Mat.ones(channels[2].rows(), channels[2].cols(), CV_16UC1), bScalar)
    merge(channels, outputMat)

    return outputMat
}

/**
 * Converting values from 16-bit to 8-bit, i.e. changing range from 0-65355 to 0-255
 */
fun convertFrom16bitTo8bit(matToConvert: Mat): Mat {
    var intermediateMat = Mat(matToConvert.cols(), matToConvert.rows(), CV_8UC3)
    matToConvert.convertTo(intermediateMat, CV_8UC3)
    return intermediateMat
}
fun getBitmap(inputMat: Mat): Bitmap {

    var bmp: Bitmap = Bitmap.createBitmap(inputMat.cols(), inputMat.rows(), Bitmap.Config.ARGB_8888)

    Utils.matToBitmap(inputMat, bmp)

    return bmp
}

fun convertToXYZ(rgbMat: Mat): Mat {
    var channels = mutableListOf<Mat>()
    var outputMat = Mat()
    split(rgbMat, channels)
    var r: Double = 0.0; var g: Double = 0.0; var b: Double = 0.0
    var X: Double = 0.0; var Y: Double = 0.0; var Z: Double = 0.0
    // D65 illuminant
    for (n in 0..rgbMat.rows()-1) {
        for (m in 0..rgbMat.cols()-1) {
            b = (channels[0][n,m][0] / 255.0)
            g = (channels[1][n,m][0] / 255.0)
            r = (channels[2][n,m][0] / 255.0)

            if (r > 0.04045) r = ((r + 0.055) / 1.055).pow(2.4) else r /= 12.92
            if (g > 0.04045) g = ((g + 0.055) / 1.055).pow(2.4) else g /= 12.92
            if (b > 0.04045) b = ((b + 0.055) / 1.055).pow(2.4) else b /= 12.92

            r *= 100.0
            g *= 100.0
            b *= 100.0

            X = r * 0.4124 + g * 0.3576 + b * 0.1805
            Y = r * 0.2126 + g * 0.7152 + b * 0.0722
            Z = r * 0.0193 + g * 0.1192 + b * 0.9505
            // calculating XYZ values
            channels[0].put(n, m, X)  // X
            channels[1].put(n, m, Y) // Y
            channels[2].put(n, m, Z)  // Z
        }
    }
    merge(channels, outputMat)

    return outputMat
}
fun convertToLab(xyzMat: Mat): Mat {
    var channels = mutableListOf<Mat>(Mat())
    var outputMat = Mat()
    var L = 0.0; var a = 0.0; var b = 0.0
    split(xyzMat, channels)

    for (n in 0..xyzMat.rows()-1) {
        for (m in 0..xyzMat.cols()-1) {
            //XYZ values
            var X = xyzMat[n,m][0]
            var Y = xyzMat[n,m][1]
            var Z = xyzMat[n,m][2]

            // converting to Lab
            if (X > 0.008856) X = X.pow(1.0/3.0) else X = (7.787 * X) + ( 16.0 / 116.0)
            if (Y > 0.008856) Y = Y.pow(1.0/3.0) else Y = (7.787 * Y) + ( 16.0 / 116.0)
            if (Z > 0.008856) Z = Z.pow(1.0/3.0) else Z = (7.787 * Z) + ( 16.0 / 116.0)

            L = (116.0 * Y) - 16 // L*
            a = 500 * (X - Y)    // a*
            b = 200 * (Y - Z)    // b*

            // calculating L*-, a*- and b*-values
            channels[0].put(n, m, L)
            channels[1].put(n, m, a)
            channels[2].put(n, m, b)
        }
    }
    merge(channels, outputMat)

    return outputMat
}