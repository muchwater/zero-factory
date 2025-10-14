'use client'

import { PlusIcon } from './IconComponents'

interface MapControlsProps {
  onZoomIn?: () => void
  onZoomOut?: () => void
  onCurrentLocation?: () => void
}

export default function MapControls({ 
  onZoomIn, 
  onZoomOut, 
  onCurrentLocation 
}: MapControlsProps) {
  return (
    <div className="absolute top-4 right-4 z-10 flex flex-col space-y-2">
      {/* 줌 인 버튼 */}
      <button
        onClick={onZoomIn}
        className="
          w-10 h-10 bg-white rounded-full shadow-md
          flex items-center justify-center
          hover:bg-gray-50 hover:shadow-lg
          transition-all duration-300 ease-in-out
          hover:scale-110
          border border-gray-200
        "
      >
        <PlusIcon className="w-5 h-5 text-gray-800" />
      </button>
      
      {/* 줌 아웃 버튼 */}
      <button
        onClick={onZoomOut}
        className="
          w-10 h-10 bg-white rounded-full shadow-md
          flex items-center justify-center
          hover:bg-gray-50 hover:shadow-lg
          transition-all duration-300 ease-in-out
          hover:scale-110
          border border-gray-200
        "
      >
        <svg className="w-5 h-5 text-gray-800" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <line x1="5" y1="12" x2="19" y2="12"/>
        </svg>
      </button>
      
      {/* 현재 위치 버튼 */}
      <button
        onClick={onCurrentLocation}
        className="
          w-10 h-10 bg-white rounded-full shadow-md
          flex items-center justify-center
          hover:bg-gray-50 hover:shadow-lg
          transition-all duration-300 ease-in-out
          hover:scale-110
          border border-gray-200
        "
      >
        <svg className="w-5 h-5 text-gray-800" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <path d="M12 2L13.09 8.26L22 9L13.09 9.74L12 16L10.91 9.74L2 9L10.91 8.26L12 2Z"/>
        </svg>
      </button>
    </div>
  )
}
