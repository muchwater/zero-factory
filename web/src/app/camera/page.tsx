'use client'

import { useState } from 'react'
import { useRouter } from 'next/navigation'
import { CameraCapture } from '@/components/CameraCapture'
import BottomNavigation from '@/components/BottomNavigation'

export default function CameraPage() {
  const router = useRouter()
  const [activeTab, setActiveTab] = useState<'home' | 'search' | 'camera' | 'profile'>('camera')

  const handleTabChange = (tab: 'home' | 'search' | 'camera' | 'profile') => {
    setActiveTab(tab)
    
    // 탭에 따라 페이지 이동
    switch (tab) {
      case 'home':
        router.push('/')
        break
      case 'search':
        router.push('/')
        break
      case 'camera':
        router.push('/camera')
        break
      case 'profile':
        router.push('/')
        break
    }
  }

  return (
    <div className="bg-background min-h-screen flex flex-col pb-20">
      {/* Header */}
      <div className="bg-white border-b border-border sticky top-0 z-40">
        <div className="px-4 py-3">
          <h1 className="text-xl font-bold text-foreground text-center">재사용 용기 인증</h1>
          <p className="text-sm text-muted text-center mt-1">카메라로 재사용 용기를 촬영하세요</p>
        </div>
      </div>

      {/* Camera Capture Section */}
      <div className="flex-1 overflow-y-auto">
        <CameraCapture />
      </div>

      {/* Bottom Navigation */}
      <BottomNavigation 
        activeTab={activeTab}
        onTabChange={handleTabChange}
      />
    </div>
  )
}
