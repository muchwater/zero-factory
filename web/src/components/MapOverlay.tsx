'use client'

import { FilterIcon, PlusIcon } from './IconComponents'

interface MapOverlayProps {
  onFilterClick?: () => void
  onAddPlaceClick?: () => void
}

export default function MapOverlay({ onFilterClick, onAddPlaceClick }: MapOverlayProps) {
  return (
    <div className="absolute top-4 left-4 right-4 z-10 flex justify-between items-start">
      {/* 필터 버튼 */}
      <button
        onClick={onFilterClick}
        className="
          flex items-center space-x-2 px-3 py-2
          bg-gray-100/95 backdrop-blur-sm rounded-lg
          border border-gray-200 shadow-md
          hover:bg-gray-200/95 hover:shadow-lg
          transition-all duration-300 ease-in-out
          hover:scale-105
        "
      >
        <FilterIcon className="w-4 h-4 text-gray-700" />
        <span className="text-sm font-medium text-gray-700">filter</span>
      </button>
      
      {/* 장소 추가 버튼 */}
      <button
        onClick={onAddPlaceClick}
        className="
          flex items-center space-x-2 px-3 py-2
          bg-gray-100/95 backdrop-blur-sm rounded-lg
          border border-gray-200 shadow-md
          hover:bg-gray-200/95 hover:shadow-lg
          transition-all duration-300 ease-in-out
          hover:scale-105
        "
      >
        <svg className="w-4 h-4 text-gray-700" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z"/>
          <circle cx="12" cy="10" r="3"/>
        </svg>
        <span className="text-sm font-medium text-gray-700">Add Places</span>
      </button>
    </div>
  )
}
