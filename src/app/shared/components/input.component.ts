import { ChangeDetectionStrategy, Component, computed, input, model } from '@angular/core';
import { FormsModule } from '@angular/forms';

@Component({
  selector: 'app-input',
  standalone: true,
  changeDetection: ChangeDetectionStrategy.OnPush,
  imports: [FormsModule],
  template: `
    <div class="w-full">
      @if (label()) {
        <label [for]="controlId" class="label-mono mb-2 block">{{ label() }}</label>
      }

      @if (type() === 'textarea') {
        <textarea
          [id]="controlId"
          [rows]="rows()"
          [placeholder]="placeholder()"
          [ngModel]="value()"
          (ngModelChange)="value.set($event)"
          [disabled]="disabled()"
          [attr.autocomplete]="autocomplete() || null"
          [class]="controlClasses()"
        ></textarea>
      } @else {
        <input
          [id]="controlId"
          [type]="type()"
          [placeholder]="placeholder()"
          [ngModel]="value()"
          (ngModelChange)="value.set($event)"
          [disabled]="disabled()"
          [attr.autocomplete]="autocomplete() || null"
          [attr.inputmode]="inputMode() || null"
          [attr.min]="min() ?? null"
          [attr.max]="max() ?? null"
          [attr.maxlength]="maxlength() ?? null"
          [class]="controlClasses()"
        />
      }

      @if (helperText()) {
        <p class="mt-2 font-mono text-[10px] leading-relaxed text-muted">{{ helperText() }}</p>
      }
    </div>
  `,
})
export class InputComponent {
  readonly label = input<string>('');
  readonly placeholder = input<string>('');
  readonly type = input<'text' | 'email' | 'password' | 'number' | 'search' | 'textarea'>('text');
  readonly disabled = input<boolean>(false);
  readonly helperText = input<string>('');
  readonly autocomplete = input<string>('');
  readonly inputMode = input<string>('');
  readonly rows = input<number>(3);
  readonly min = input<number | null>(null);
  readonly max = input<number | null>(null);
  readonly maxlength = input<number | null>(null);
  readonly id = input<string>('');
  readonly value = model<string>('');

  private static nextId = 0;
  private readonly fallbackId = `field-${InputComponent.nextId++}`;

  get controlId(): string {
    return this.id() || this.fallbackId;
  }

  readonly controlClasses = computed(() => {
    const base =
      'command-input w-full disabled:cursor-not-allowed disabled:opacity-50';
    return this.type() === 'textarea' ? `${base} resize-none leading-relaxed` : base;
  });
}
