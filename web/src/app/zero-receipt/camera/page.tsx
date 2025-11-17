'use client'

import { useState, useRef, useEffect } from 'react'
import { useRouter } from 'next/navigation'

export default function CameraPage() {
  const router = useRouter()
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [stream, setStream] = useState<MediaStream | null>(null)
  const [facingMode, setFacingMode] = useState<'user' | 'environment'>('environment')
  const [capturedImage, setCapturedImage] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    startCamera()
    return () => {
      stopCamera()
    }
  }, [facingMode])

  const startCamera = async () => {
    try {
      const mediaStream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: facingMode,
          width: { ideal: 1280 },
          height: { ideal: 720 }
        }
      })
      
      if (videoRef.current) {
        videoRef.current.srcObject = mediaStream
        setStream(mediaStream)
        setError(null)
      }
    } catch (err) {
      console.error('카메라 접근 오류:', err)
      setError('카메라에 접근할 수 없습니다. 브라우저 권한을 확인해주세요.')
    }
  }

  const stopCamera = () => {
    if (stream) {
      stream.getTracks().forEach(track => track.stop())
      setStream(null)
    }
  }

  const switchCamera = () => {
    setFacingMode(prev => prev === 'user' ? 'environment' : 'user')
  }

  const capturePhoto = () => {
    if (videoRef.current && canvasRef.current) {
      const video = videoRef.current
      const canvas = canvasRef.current
      const ctx = canvas.getContext('2d')

      if (ctx) {
        canvas.width = video.videoWidth
        canvas.height = video.videoHeight
        ctx.drawImage(video, 0, 0)
        
        const imageDataUrl = canvas.toDataURL('image/jpeg')
        setCapturedImage(imageDataUrl)
        stopCamera()
      }
    }
  }

  const retakePhoto = () => {
    setCapturedImage(null)
    startCamera()
  }

  const usePhoto = () => {
    if (capturedImage) {
      // 제로영수증 페이지로 돌아가면서 이미지 전달
      router.push(`/zero-receipt?photo=${encodeURIComponent(capturedImage)}`)
    }
  }

  const handleBack = () => {
    stopCamera()
    router.push('/zero-receipt')
  }

  if (error) {
    return (
      <div className="bg-black min-h-screen flex flex-col items-center justify-center text-white p-4">
        <div className="text-center">
          <p className="text-lg mb-4">{error}</p>
          <button
            onClick={handleBack}
            className="px-6 py-2 bg-white text-black rounded-lg"
          >
            돌아가기
          </button>
        </div>
      </div>
    )
  }

  if (capturedImage) {
    return (
      <div className="bg-black min-h-screen flex flex-col">
        {/* Header */}
        <div className="bg-white/10 backdrop-blur-sm border-b border-white/20 sticky top-0 z-40">
          <div className="px-4 py-3 flex items-center justify-between">
            <button
              onClick={retakePhoto}
              className="text-white text-lg font-medium"
            >
              다시 촬영
            </button>
            <h1 className="text-xl font-bold text-white">사진 확인</h1>
            <div className="w-16" /> {/* Spacer */}
          </div>
        </div>

        {/* Captured Image */}
        <div className="flex-1 flex items-center justify-center p-4">
          <img
            src={capturedImage}
            alt="Captured"
            className="max-w-full max-h-full rounded-lg"
          />
        </div>

        {/* Action Buttons */}
        <div className="bg-gradient-to-t from-black/90 to-transparent p-6 pb-12">
          <div className="flex gap-4 justify-center">
            <button
              onClick={retakePhoto}
              className="px-6 py-3 bg-white/20 text-white rounded-xl font-medium"
            >
              다시 촬영
            </button>
            <button
              onClick={usePhoto}
              className="px-6 py-3 bg-primary text-white rounded-xl font-bold"
            >
              이 사진 사용
            </button>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="bg-black min-h-screen flex flex-col relative">
      {/* Header */}
      <div className="bg-white/10 backdrop-blur-sm border-b border-white/20 sticky top-0 z-40">
        <div className="px-4 py-3 flex items-center justify-between">
          <button
            onClick={handleBack}
            className="text-white"
          >
            <svg
              className="w-6 h-6"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M15 19l-7-7 7-7"
              />
            </svg>
          </button>
          <h1 className="text-xl font-bold text-white">촬영하기</h1>
          <button
            onClick={switchCamera}
            className="text-white"
          >
            <svg
              className="w-6 h-6"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"
              />
            </svg>
          </button>
        </div>
      </div>

      {/* Camera View */}
      <div className="flex-1 relative overflow-hidden">
        <video
          ref={videoRef}
          autoPlay
          playsInline
          className="w-full h-full object-cover"
        />
        
        {/* Framing Guides */}
        <div className="absolute inset-0 pointer-events-none">
          {/* Corner guides */}
          <div className="absolute top-8 left-8 w-8 h-8 border-t-2 border-l-2 border-white" />
          <div className="absolute top-8 right-8 w-8 h-8 border-t-2 border-r-2 border-white" />
          <div className="absolute bottom-32 left-8 w-8 h-8 border-b-2 border-l-2 border-white" />
          <div className="absolute bottom-32 right-8 w-8 h-8 border-b-2 border-r-2 border-white" />
        </div>

        {/* Instruction Text */}
        <div className="absolute bottom-40 left-0 right-0 text-center">
          <p className="text-red-500 font-bold text-base">
            조금만 더 기울여주세요!
          </p>
        </div>
      </div>

      {/* Bottom Controls */}
      <div className="bg-gradient-to-t from-black/90 to-transparent p-6 pb-12">
        <div className="flex items-center justify-between">
          {/* Menu Button */}
          <button className="w-12 h-12 flex items-center justify-center">
            <svg
              className="w-6 h-6 text-white"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M12 5v.01M12 12v.01M12 19v.01M12 6a1 1 0 110-2 1 1 0 010 2zm0 7a1 1 0 110-2 1 1 0 010 2zm0 7a1 1 0 110-2 1 1 0 010 2z"
              />
            </svg>
          </button>

          {/* Shutter Button */}
          <button
            onClick={capturePhoto}
            className="w-20 h-20 rounded-full bg-white border-4 border-gray-300 flex items-center justify-center shadow-lg active:scale-95 transition-transform"
          >
            <div className="w-16 h-16 rounded-full bg-white" />
          </button>

          {/* Rotate Button */}
          <button
            onClick={switchCamera}
            className="w-12 h-12 flex items-center justify-center"
          >
            <svg
              className="w-6 h-6 text-white"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"
              />
            </svg>
          </button>
        </div>
      </div>

      {/* Hidden canvas for capturing */}
      <canvas ref={canvasRef} className="hidden" />
    </div>
  )
}

