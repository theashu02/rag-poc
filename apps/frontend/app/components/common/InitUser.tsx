'use client';

import { useEffect } from 'react';
import { getUser } from '@/lib/ApiStore/actions/getCurrentUser';
import { useUserStore } from '@/store/useUserStore';

export default function InitUser() {
  const setUser = useUserStore((s) => s.setUser);
  useEffect(() => {
    getUser().then(user => {
      if (user) setUser(user);
    });
  }, [setUser]);
  return null;
}