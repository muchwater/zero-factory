'use client'

import { useEffect, useState } from 'react'

interface EnvCheckerProps {
  children: React.ReactNode
}

export default function EnvChecker({ children }: EnvCheckerProps) {
  const [envStatus, setEnvStatus] = useState<{
    hasApiKey: boolean
    apiKeyValue: string | null
    error: string | null
  }>({
    hasApiKey: false,
    apiKeyValue: null,
    error: null
  })

  useEffect(() => {
    const apiKey = process.env.NEXT_PUBLIC_KAKAO_MAP_API_KEY
    
    if (!apiKey) {
      setEnvStatus({
        hasApiKey: false,
        apiKeyValue: null,
        error: 'NEXT_PUBLIC_KAKAO_MAP_API_KEY 환경변수가 설정되지 않았습니다.'
      })
      return
    }

    if (apiKey === 'YOUR_API_KEY' || apiKey.length < 10) {
      setEnvStatus({
        hasApiKey: false,
        apiKeyValue: apiKey,
        error: '유효하지 않은 카카오맵 API 키입니다.'
      })
      return
    }

    setEnvStatus({
      hasApiKey: true,
      apiKeyValue: apiKey,
      error: null
    })
  }, [])

  // 개발 환경에서만 에러 표시
  if (process.env.NODE_ENV === 'development' && envStatus.error) {
    return (
      <div className="min-h-screen bg-red-50 flex items-center justify-center p-4">
        <div className="bg-white rounded-lg shadow-lg p-6 max-w-md w-full">
          <div className="text-center">
            <div className="text-red-500 text-4xl mb-4">⚠️</div>
            <h2 className="text-xl font-bold text-gray-800 mb-2">
              카카오맵 설정 오류
            </h2>
            <p className="text-gray-600 mb-4">
              {envStatus.error}
            </p>
            
            <div className="bg-gray-100 rounded p-4 text-left">
              <h3 className="font-medium text-gray-700 mb-2">해결 방법:</h3>
              <ol className="text-sm text-gray-600 space-y-1">
                <li>1. 프로젝트 루트의 .env.local 파일 확인</li>
                <li>2. NEXT_PUBLIC_KAKAO_MAP_API_KEY=실제_API_키 설정</li>
                <li>3. 개발 서버 재시작 (npm run dev)</li>
                <li>4. 카카오 개발자 사이트에서 도메인 등록 확인</li>
              </ol>
            </div>

            {envStatus.apiKeyValue && (
              <div className="mt-4 p-3 bg-yellow-100 rounded text-left">
                <p className="text-sm text-yellow-800">
                  <strong>현재 설정된 값:</strong> {envStatus.apiKeyValue}
                </p>
              </div>
            )}
          </div>
        </div>
      </div>
    )
  }

  return <>{children}</>
}
