'use client'

import { useState, useEffect } from 'react'
import { SearchIcon } from './IconComponents'

interface SearchBarProps {
  placeholder?: string
  onSearch?: (value: string) => void
  className?: string
  value?: string
  suggestions?: string[]
  onSuggestionClick?: (suggestion: string) => void
}

export default function SearchBar({ 
  placeholder = "상점명 또는 지역명으로 검색...", 
  onSearch,
  className = "",
  value: controlledValue,
  suggestions = [],
  onSuggestionClick
}: SearchBarProps) {
  const [searchValue, setSearchValue] = useState(controlledValue || '')
  const [showSuggestions, setShowSuggestions] = useState(false)
  const [isFocused, setIsFocused] = useState(false)
  
  // controlledValue가 변경되면 내부 상태 업데이트
  useEffect(() => {
    if (controlledValue !== undefined) {
      setSearchValue(controlledValue)
    }
  }, [controlledValue])

  // 검색어가 있고 포커스가 있을 때만 연관 검색어 표시
  useEffect(() => {
    setShowSuggestions(isFocused && searchValue.trim().length > 0 && suggestions.length > 0)
  }, [isFocused, searchValue, suggestions])

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    onSearch?.(searchValue)
  }

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = e.target.value
    setSearchValue(value)
    // 실시간 검색
    onSearch?.(value)
  }

  const handleSuggestionClick = (suggestion: string) => {
    setSearchValue(suggestion)
    setShowSuggestions(false)
    onSuggestionClick?.(suggestion)
    onSearch?.(suggestion)
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
            onFocus={() => setIsFocused(true)}
            onBlur={() => {
              // 클릭 이벤트가 먼저 처리되도록 약간의 지연
              setTimeout(() => setIsFocused(false), 200)
            }}
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

        {/* 연관 검색어 드롭다운 */}
        {showSuggestions && suggestions.length > 0 && (
          <div className="absolute top-full left-0 right-0 z-50 bg-white border border-gray-200 rounded-xl shadow-lg mt-1 max-h-60 overflow-y-auto">
            {suggestions.map((suggestion, index) => (
              <button
                key={index}
                type="button"
                onClick={() => handleSuggestionClick(suggestion)}
                className="w-full px-4 py-3 text-left text-sm text-gray-700 hover:bg-gray-50 transition-colors border-b border-gray-100 last:border-b-0"
              >
                <div className="flex items-center gap-2">
                  <SearchIcon className="w-4 h-4 text-gray-400" />
                  <span>{suggestion}</span>
                </div>
              </button>
            ))}
          </div>
        )}
      </form>
    </div>
  )
}
