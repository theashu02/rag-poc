'use client';

import { useEffect } from 'react';
import { useDispatch } from 'react-redux';                
import { getUser } from '@/lib/ApiStore/actions/getCurrentUser';
import { setUser } from '@/store/slices/UserStoreSlice';       
import type { AppDispatch } from '@/store/store';       

export default function InitUser() {
  const dispatch = useDispatch<AppDispatch>();

  useEffect(() => {
    getUser().then(user => {
      if (user) dispatch(setUser(user));                 
    });
  }, [dispatch]);

  return null;
}