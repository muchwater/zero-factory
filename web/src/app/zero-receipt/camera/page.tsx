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
  const [isVerifying, setIsVerifying] = useState(false)
  const [verificationResult, setVerificationResult] = useState<any>(null)

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

  const capturePhoto = async () => {
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

        // 자동으로 AI 검증 시작
        await verifyContainer(imageDataUrl)
      }
    }
  }

  const verifyContainer = async (imageDataUrl: string) => {
    setIsVerifying(true)
    setError(null)

    try {
      // Data URL을 Blob으로 변환
      const response = await fetch(imageDataUrl)
      const blob = await response.blob()

      // FormData 생성
      const formData = new FormData()
      formData.append('file', blob, 'photo.jpg')

      // AI API 호출 (프로덕션: https://ai.zeromap.store, 개발: http://localhost:8000)
      const AI_API_URL = typeof window !== 'undefined' && window.location.hostname === 'localhost'
        ? 'http://localhost:8000'
        : 'https://ai.zeromap.store'
      const aiResponse = await fetch(`${AI_API_URL}/container/verify`, {
        method: 'POST',
        body: formData,
      })

      if (!aiResponse.ok) {
        throw new Error('AI 검증에 실패했습니다')
      }

      const result = await aiResponse.json()
      setVerificationResult(result)
    } catch (err) {
      console.error('AI 검증 오류:', err)
      setError('AI 검증 중 오류가 발생했습니다. 다시 시도해주세요.')
    } finally {
      setIsVerifying(false)
    }
  }

  const retakePhoto = () => {
    setCapturedImage(null)
    setVerificationResult(null)
    setError(null)
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
      <div className="bg-black h-screen flex flex-col overflow-hidden">
        {/* Header */}
        <div className="bg-white/10 backdrop-blur-sm border-b border-white/20 z-40 flex-shrink-0">
          <div className="px-4 py-3 flex items-center justify-between">
            <button
              onClick={retakePhoto}
              className="text-white text-sm font-medium"
            >
              다시 촬영
            </button>
            <h1 className="text-lg font-bold text-white">
              {isVerifying ? 'AI 검증 중...' : verificationResult ? 'AI 검증 결과' : '사진 확인'}
            </h1>
            <div className="w-16" /> {/* Spacer */}
          </div>
        </div>

        {/* Content Area - Scrollable */}
        <div className="flex-1 overflow-y-auto">
          <div className="flex flex-col items-center p-4 pb-24">
            {/* Captured Image */}
            <img
              src={capturedImage}
              alt="Captured"
              className="w-full max-w-sm rounded-lg mb-4 object-contain"
              style={{ maxHeight: '40vh' }}
            />

            {/* Loading State */}
            {isVerifying && (
              <div className="bg-white/10 backdrop-blur-sm rounded-xl p-6 text-center w-full max-w-sm">
                <div className="animate-spin w-12 h-12 border-4 border-primary border-t-transparent rounded-full mx-auto mb-4"></div>
                <p className="text-white font-medium">AI가 검증하는 중입니다...</p>
                <p className="text-white/60 text-sm mt-2">잠시만 기다려주세요</p>
              </div>
            )}

            {/* Verification Result */}
            {!isVerifying && verificationResult && (
              <div className="bg-white/10 backdrop-blur-sm rounded-xl p-6 w-full max-w-sm">
                {/* Success: Reusable Container with Beverage */}
                {verificationResult.is_reusable && verificationResult.beverage_status === 'Yes' && (
                  <div className="text-center">
                    <div className="text-6xl mb-4">✅</div>
                    <h2 className="text-xl font-bold text-green-400 mb-2">검증 성공!</h2>
                    <p className="text-white/80 mb-4 text-sm">다회용기에 음료가 담긴 것을 확인했습니다</p>
                    <div className="space-y-2 text-xs text-white/60">
                      <p>• 다회용기 신뢰도: {(verificationResult.reusable_confidence * 100).toFixed(1)}%</p>
                      <p>• 음료 감지 신뢰도: {(verificationResult.beverage_confidence * 100).toFixed(1)}%</p>
                      {verificationResult.container_class && (
                        <p>• 용기 종류: {verificationResult.container_class === 'cup' ? '컵' : '병'}</p>
                      )}
                    </div>
                  </div>
                )}

                {/* Success: Reusable Container but No Beverage */}
                {verificationResult.is_reusable && verificationResult.beverage_status === 'No' && (
                  <div className="text-center">
                    <div className="text-5xl mb-4">⚠️</div>
                    <h2 className="text-xl font-bold text-yellow-400 mb-2">음료가 없습니다</h2>
                    <p className="text-white/80 mb-4 text-sm">다회용기이지만 음료가 담겨있지 않습니다</p>
                    <div className="space-y-2 text-xs text-white/60">
                      <p>• 다회용기 신뢰도: {(verificationResult.reusable_confidence * 100).toFixed(1)}%</p>
                      <p>• 음료 없음 신뢰도: {(verificationResult.beverage_confidence * 100).toFixed(1)}%</p>
                    </div>
                  </div>
                )}

                {/* Success: Reusable Container but Unclear Beverage */}
                {verificationResult.is_reusable && verificationResult.beverage_status === 'Unclear' && (
                  <div className="text-center">
                    <div className="text-5xl mb-4">🤔</div>
                    <h2 className="text-xl font-bold text-blue-400 mb-2">음료 확인 불가</h2>
                    <p className="text-white/80 mb-4 text-sm">다회용기이나 음료 유무를 명확히 판단할 수 없습니다</p>
                    <div className="space-y-2 text-xs text-white/60">
                      <p>• 다회용기 신뢰도: {(verificationResult.reusable_confidence * 100).toFixed(1)}%</p>
                      <p>더 선명한 사진을 다시 촬영해주세요</p>
                    </div>
                  </div>
                )}

                {/* Failure: Not Reusable */}
                {verificationResult.container_detected && !verificationResult.is_reusable && (
                  <div className="text-center">
                    <div className="text-5xl mb-4">❌</div>
                    <h2 className="text-xl font-bold text-red-400 mb-2">일회용기 감지</h2>
                    <p className="text-white/80 mb-4 text-sm">일회용 용기로 판단됩니다</p>
                    <div className="space-y-2 text-xs text-white/60">
                      <p>• 일회용기 신뢰도: {(verificationResult.reusable_confidence * 100).toFixed(1)}%</p>
                      <p>다회용기를 사용해주세요</p>
                    </div>
                  </div>
                )}

                {/* Failure: No Container Detected */}
                {!verificationResult.container_detected && (
                  <div className="text-center">
                    <div className="text-5xl mb-4">🔍</div>
                    <h2 className="text-xl font-bold text-red-400 mb-2">용기를 찾을 수 없습니다</h2>
                    <p className="text-white/80 mb-4 text-sm">사진에서 용기가 감지되지 않았습니다</p>
                    <p className="text-xs text-white/60">컵이나 병을 더 선명하게 촬영해주세요</p>
                  </div>
                )}
              </div>
            )}

            {/* Error State */}
            {!isVerifying && error && (
              <div className="bg-red-500/20 backdrop-blur-sm rounded-xl p-6 w-full max-w-sm">
                <div className="text-center">
                  <div className="text-5xl mb-4">⚠️</div>
                  <h2 className="text-xl font-bold text-red-400 mb-2">오류 발생</h2>
                  <p className="text-white/80 text-sm">{error}</p>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Action Buttons - Fixed at bottom */}
        <div className="bg-gradient-to-t from-black via-black/95 to-transparent p-4 flex-shrink-0 absolute bottom-0 left-0 right-0">
          <div className="flex gap-3 justify-center max-w-md mx-auto">
            <button
              onClick={retakePhoto}
              className="flex-1 px-4 py-3 bg-white/20 text-white rounded-xl font-medium text-sm active:bg-white/30"
            >
              다시 촬영
            </button>
            {verificationResult?.is_reusable && verificationResult?.beverage_status === 'Yes' && (
              <button
                onClick={usePhoto}
                className="flex-1 px-4 py-3 bg-green-500 text-white rounded-xl font-bold text-sm active:bg-green-600"
              >
                이 사진 사용
              </button>
            )}
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="bg-black h-screen flex flex-col relative overflow-hidden">
      {/* Header */}
      <div className="bg-white/10 backdrop-blur-sm border-b border-white/20 z-40 flex-shrink-0">
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
          <div className="absolute top-4 left-4 w-6 h-6 border-t-2 border-l-2 border-white" />
          <div className="absolute top-4 right-4 w-6 h-6 border-t-2 border-r-2 border-white" />
          <div className="absolute bottom-24 left-4 w-6 h-6 border-b-2 border-l-2 border-white" />
          <div className="absolute bottom-24 right-4 w-6 h-6 border-b-2 border-r-2 border-white" />
        </div>

        {/* Instruction Text */}
        <div className="absolute bottom-32 left-0 right-0 text-center px-4">
          <p className="text-white font-bold text-base bg-black/50 py-2 px-4 rounded-lg inline-block">
            컵 내부가 보이게 찍어주세요
          </p>
        </div>
      </div>

      {/* Bottom Controls - Fixed height and positioning */}
      <div className="bg-gradient-to-t from-black/90 to-transparent p-4 pb-6 flex-shrink-0">
        <div className="flex items-center justify-between max-w-md mx-auto">
          {/* Menu Button - Hidden for cleaner UI */}
          <div className="w-12 h-12" />

          {/* Shutter Button */}
          <button
            onClick={capturePhoto}
            className="w-16 h-16 rounded-full bg-white border-4 border-gray-300 flex items-center justify-center shadow-lg active:scale-95 transition-transform"
          >
            <div className="w-12 h-12 rounded-full bg-white" />
          </button>

          {/* Rotate Button */}
          <button
            onClick={switchCamera}
            className="w-12 h-12 flex items-center justify-center bg-white/10 rounded-full active:bg-white/20"
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

