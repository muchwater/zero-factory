"use client"

import { useState, useRef, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import { Camera, RotateCcw, Download } from "lucide-react"

interface SensorData {
  alpha: number | null // Z축 회전 (0-360도)
  beta: number | null // X축 회전 (-180~180도)
  gamma: number | null // Y축 회전 (-90~90도)
  accelerationX: number | null
  accelerationY: number | null
  accelerationZ: number | null
}

export function CameraCapture() {
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [stream, setStream] = useState<MediaStream | null>(null)
  const [capturedImage, setCapturedImage] = useState<string | null>(null)
  const [sensorData, setSensorData] = useState<SensorData>({
    alpha: null,
    beta: null,
    gamma: null,
    accelerationX: null,
    accelerationY: null,
    accelerationZ: null,
  })
  const [isPermissionGranted, setIsPermissionGranted] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    // DeviceOrientation 이벤트 리스너
    const handleOrientation = (event: DeviceOrientationEvent) => {
      setSensorData((prev) => ({
        ...prev,
        alpha: event.alpha,
        beta: event.beta,
        gamma: event.gamma,
      }))
    }

    // DeviceMotion 이벤트 리스너
    const handleMotion = (event: DeviceMotionEvent) => {
      if (event.accelerationIncludingGravity) {
        setSensorData((prev) => ({
          ...prev,
          accelerationX: event.accelerationIncludingGravity?.x || null,
          accelerationY: event.accelerationIncludingGravity?.y || null,
          accelerationZ: event.accelerationIncludingGravity?.z || null,
        }))
      }
    }

    // iOS 13+ 권한 요청
    const requestSensorPermission = async () => {
      if (typeof (DeviceOrientationEvent as any).requestPermission === "function") {
        try {
          const permission = await (DeviceOrientationEvent as any).requestPermission()
          if (permission === "granted") {
            setIsPermissionGranted(true)
            window.addEventListener("deviceorientation", handleOrientation)
            window.addEventListener("devicemotion", handleMotion)
          }
        } catch (err) {
          console.error("센서 권한 요청 실패:", err)
        }
      } else {
        // iOS 13 미만 또는 Android
        setIsPermissionGranted(true)
        window.addEventListener("deviceorientation", handleOrientation)
        window.addEventListener("devicemotion", handleMotion)
      }
    }

    requestSensorPermission()

    return () => {
      window.removeEventListener("deviceorientation", handleOrientation)
      window.removeEventListener("devicemotion", handleMotion)
    }
  }, [])

  const startCamera = async () => {
    try {
      setError(null)
      const mediaStream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: "environment", // 후면 카메라
          width: { ideal: 1920 },
          height: { ideal: 1080 },
        },
      })
      setStream(mediaStream)
      if (videoRef.current) {
        videoRef.current.srcObject = mediaStream
      }
    } catch (err) {
      setError("카메라 접근 권한이 필요합니다.")
      console.error("카메라 시작 실패:", err)
    }
  }

  const stopCamera = () => {
    if (stream) {
      stream.getTracks().forEach((track) => track.stop())
      setStream(null)
    }
  }

  const capturePhoto = () => {
    if (videoRef.current && canvasRef.current) {
      const video = videoRef.current
      const canvas = canvasRef.current
      canvas.width = video.videoWidth
      canvas.height = video.videoHeight

      const context = canvas.getContext("2d")
      if (context) {
        context.drawImage(video, 0, 0, canvas.width, canvas.height)
        const imageData = canvas.toDataURL("image/jpeg", 0.95)
        setCapturedImage(imageData)

        console.log("[v0] Photo captured with sensor data:", sensorData)
      }
    }
  }

  const downloadPhoto = () => {
    if (capturedImage) {
      const link = document.createElement("a")
      link.href = capturedImage
      link.download = `photo-${Date.now()}.jpg`
      link.click()
    }
  }

  const retake = () => {
    setCapturedImage(null)
  }

  useEffect(() => {
    return () => {
      stopCamera()
    }
  }, [])

  return (
    <div className="w-full max-w-4xl mx-auto p-4 space-y-4">
      <Card>
        <CardContent className="p-6">
          <div className="aspect-video bg-muted rounded-lg overflow-hidden relative">
            {!stream && !capturedImage && (
              <div className="absolute inset-0 flex items-center justify-center">
                <Button onClick={startCamera} size="lg">
                  <Camera className="mr-2 h-5 w-5" />
                  카메라 시작
                </Button>
              </div>
            )}

            {stream && !capturedImage && (
              <>
                <video ref={videoRef} autoPlay playsInline className="w-full h-full object-cover" />
                <div className="absolute bottom-4 left-1/2 -translate-x-1/2">
                  <Button onClick={capturePhoto} size="lg" className="rounded-full h-16 w-16">
                    <Camera className="h-6 w-6" />
                  </Button>
                </div>
              </>
            )}

            {capturedImage && (
              <>
                <img src={capturedImage || "/placeholder.svg"} alt="Captured" className="w-full h-full object-cover" />
                <div className="absolute bottom-4 left-1/2 -translate-x-1/2 flex gap-2">
                  <Button onClick={retake} size="lg" variant="secondary">
                    <RotateCcw className="mr-2 h-5 w-5" />
                    다시 찍기
                  </Button>
                  <Button onClick={downloadPhoto} size="lg">
                    <Download className="mr-2 h-5 w-5" />
                    다운로드
                  </Button>
                </div>
              </>
            )}

            <canvas ref={canvasRef} className="hidden" />
          </div>

          {error && <p className="text-destructive text-sm mt-4">{error}</p>}
        </CardContent>
      </Card>

      {/* IMU 센서 데이터 표시 */}
      <Card>
        <CardContent className="p-6">
          <h3 className="text-lg font-semibold mb-4">IMU 센서 데이터</h3>
          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-2">
              <h4 className="text-sm font-medium text-muted-foreground">방향 (Orientation)</h4>
              <div className="space-y-1 font-mono text-sm">
                <p>Alpha (Z축): {sensorData.alpha?.toFixed(2) ?? "N/A"}°</p>
                <p>Beta (X축): {sensorData.beta?.toFixed(2) ?? "N/A"}°</p>
                <p>Gamma (Y축): {sensorData.gamma?.toFixed(2) ?? "N/A"}°</p>
              </div>
            </div>
            <div className="space-y-2">
              <h4 className="text-sm font-medium text-muted-foreground">가속도 (Acceleration)</h4>
              <div className="space-y-1 font-mono text-sm">
                <p>X: {sensorData.accelerationX?.toFixed(2) ?? "N/A"} m/s²</p>
                <p>Y: {sensorData.accelerationY?.toFixed(2) ?? "N/A"} m/s²</p>
                <p>Z: {sensorData.accelerationZ?.toFixed(2) ?? "N/A"} m/s²</p>
              </div>
            </div>
          </div>

          {!isPermissionGranted && (
            <p className="text-muted-foreground text-sm mt-4">
              💡 센서 데이터를 보려면 디바이스 권한이 필요합니다. (iOS의 경우 HTTPS 필요)
            </p>
          )}
        </CardContent>
      </Card>

      {/* 기울기 시각화 */}
      {sensorData.beta !== null && sensorData.gamma !== null && (
        <Card>
          <CardContent className="p-6">
            <h3 className="text-lg font-semibold mb-4">기울기 시각화</h3>
            <div className="flex items-center justify-center h-64 bg-muted rounded-lg relative overflow-hidden">
              <div
                className="w-32 h-32 bg-primary rounded-full transition-transform duration-100"
                style={{
                  transform: `translateX(${(sensorData.gamma || 0) * 2}px) translateY(${(sensorData.beta || 0) * 2}px)`,
                }}
              />
              <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
                <div className="w-px h-full bg-border" />
                <div className="w-full h-px bg-border absolute" />
              </div>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  )
}
