<button (click)="theme.toggleTheme()" class="p-2 transition-colors">
     <ng-container *ngIf="theme.isDarkMode; else lightIcon">
       <i class="ri-moon-line"></i>
     </ng-container>
     <ng-template #lightIcon>
       <i class="ri-sun-line"></i>
     </ng-template>
   </button>