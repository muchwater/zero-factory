'use client'

import { useState } from 'react'
import { SearchIcon } from './IconComponents'

interface SearchBarProps {
  placeholder?: string
  onSearch?: (value: string) => void
  className?: string
}

export default function SearchBar({ 
  placeholder = "상점명 또는 지역명으로 검색...", 
  onSearch,
  className = ""
}: SearchBarProps) {
  const [searchValue, setSearchValue] = useState('')

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    onSearch?.(searchValue)
  }

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = e.target.value
    setSearchValue(value)
    // 실시간 검색을 원한다면 여기서 onSearch 호출
    // onSearch?.(value)
  }

  return (
    <div className={`w-full ${className}`}>
      <form onSubmit={handleSubmit} className="relative">
        <div className="relative">
          <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
            <SearchIcon className="w-5 h-5 text-muted" />
          </div>
          
          <input
            type="text"
            value={searchValue}
            onChange={handleInputChange}
            placeholder={placeholder}
            className="
              w-full pl-10 pr-4 py-3 
              border border-border rounded-xl 
              bg-white text-foreground placeholder-muted
              focus:outline-none focus:ring-2 focus:ring-primary/20 focus:border-primary
              transition-all duration-300 ease-in-out
              shadow-sm hover:shadow-md focus:shadow-lg
            "
          />
          
          {searchValue && (
            <button
              type="button"
              onClick={() => {
                setSearchValue('')
                onSearch?.('')
              }}
              className="absolute inset-y-0 right-0 pr-3 flex items-center text-muted hover:text-foreground transition-colors duration-200"
            >
              <svg className="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <line x1="18" y1="6" x2="6" y2="18"/>
                <line x1="6" y1="6" x2="18" y2="18"/>
              </svg>
            </button>
          )}
        </div>
      </form>
    </div>
  )
}
